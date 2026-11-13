import ray
import json
import time
# Notice we removed torch, spacy, gliner, and transformers from global imports

ray.init(ignore_reinit_error=True)

# --- 1. THE PERCEPTION ACTOR (GPU) ---
@ray.remote(num_cpus=2)
class GLiNERActor:
    def __init__(self):
        # We use natural language prompts for the zero-shot encoder
        # We will map these back to UMLS TUIs in the Rust/SHACL layer later
        self.labels = [
            "Disease",               # Maps to T047
            "Anatomy",               # Maps to T023
            "Clinical finding",      # Maps to T033 & T190
            "Measurement",           # Maps to T081
            "Modifier",              # Maps to T080
            "Substance"              # Added to catch "free air" and "fluid"
        ]
        self.model = None
        self.nlp = None
        self.device = None

    def _initialize_models(self):
        if self.model is None:
            import torch
            import spacy
            from gliner import GLiNER
            
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            # Upgraded to the v2.5 boundary-prediction architecture
            self.model = GLiNER.from_pretrained("gliner-community/gliner_base-v2.5").to(self.device)
            self.nlp = spacy.load("en_core_web_sm")

    def extract(self, report_id, text):
        self._initialize_models()
        doc = self.nlp(text)
        results = []
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text or "Study:" in sentence_text: continue
            
            entities = self.model.predict_entities(sentence_text, self.labels, threshold=0.3)
            for ent in entities:
                results.append({
                    "report_id": report_id,
                    "entity": ent["text"],
                    "type": ent["label"],
                    "source_sentence": sentence_text
                })
        return results

# --- 2. THE VERIFIER ACTOR (GPU) ---
@ray.remote(num_cpus=2)
class NLIActor:
    def __init__(self):
        self.assertion_labels = ["Present", "Absent", "Normal"]
        self.classifier = None

    def _initialize_models(self):
        if self.classifier is None:
            # HIDING IMPORTS FROM RAY SERIALIZER
            import torch
            from transformers import pipeline
            
            device_name = "mps" if torch.backends.mps.is_available() else "cpu"
            device = torch.device(device_name)
            self.classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=device
            )

    def verify(self, extracted_entities):
        self._initialize_models()
        verified_claims = []
        for item in extracted_entities:
            hypothesis = f"The {item['entity']} is {{}}."
            nli_result = self.classifier(
                item["source_sentence"],
                candidate_labels=self.assertion_labels,
                hypothesis_template=hypothesis,
                multi_label=False
            )
            item["assertion"] = nli_result['labels'][0]
            item["confidence"] = round(nli_result['scores'][0], 3)
            verified_claims.append(item)
        return verified_claims

# --- 3. THE FIREWALL ACTOR (CPU) ---
@ray.remote(num_cpus=1)
class SHACLFirewallActor:
    def __init__(self):
        # We can safely import rdflib globally or locally, but locally is safer for Ray
        from rdflib import Graph, Namespace
        self.RAD = Namespace("http://radlex.org/")
        self.EX = Namespace("http://hospital.org/clinical/")
        self.ontology_rules = """
        @prefix rad: <http://radlex.org/> .
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://hospital.org/clinical/> .
        ex:ClaimShape a sh:NodeShape ;
            sh:targetClass ex:ClinicalFact ;
            sh:property [
                sh:path rad:hasAssertion ;
                sh:minCount 1 ;
                sh:in ("Present" "Absent" "Normal") ;
            ] .
        """
        self.shacl_graph = Graph().parse(data=self.ontology_rules, format="turtle")

    def compile_and_validate(self, verified_semantic, deterministic):
        from rdflib import Graph, Literal
        from rdflib.namespace import RDF
        from pyshacl import validate
        
        all_claims = verified_semantic + deterministic
        if not all_claims:
            return {"report_id": None, "valid": False, "claims": []}
            
        report_id = all_claims[0]["report_id"]

        # --- EDGE RESOLUTION (FUSION) ---
        # Group all extractions by their source sentence
        sentences = set(c['source_sentence'] for c in all_claims)
        
        for sentence in sentences:
            nodes_in_sent = [c for c in all_claims if c['source_sentence'] == sentence]
            measurements = [n for n in nodes_in_sent if n['type'] == 'Measurement']
            targets = [n for n in nodes_in_sent if n['type'] in ['Anatomy', 'Clinical finding', 'Disease']]
            
            for meas in measurements:
                # Find the closest semantic target based on character distance
                closest_target = None
                min_distance = float('inf')
                
                for tgt in targets:
                    # Calculate gap between the measurement span and target span
                    dist = min(abs(meas['start'] - tgt['end']), abs(tgt['start'] - meas['end']))
                    if dist < min_distance:
                        min_distance = dist
                        closest_target = tgt
                
                # Create the edge in the JSON payload
                if closest_target:
                    meas['modifies_entity'] = closest_target['entity']
                    meas['modifies_type'] = closest_target['type']

        # --- SHACL GRAPH CONSTRUCTION ---
        data_graph = Graph()
        for idx, claim in enumerate(all_claims):
            fact_uri = self.EX[f"fact_{report_id}_{idx}"]
            data_graph.add((fact_uri, RDF.type, self.EX.ClinicalFact))
            data_graph.add((fact_uri, self.RAD.hasEntity, Literal(claim['entity'])))
            data_graph.add((fact_uri, self.RAD.hasAssertion, Literal(claim['assertion'])))
            
            # Write the resolved edge to the RDF graph
            if 'modifies_entity' in claim:
                # In production, link to the actual URI of the target node
                data_graph.add((fact_uri, self.RAD.modifies, Literal(claim['modifies_entity'])))
        
        conforms, _, _ = validate(data_graph, shacl_graph=self.shacl_graph)
        return {"report_id": report_id, "valid": conforms, "claims": all_claims}

# --- 4. ORCHESTRATION SCRIPT ---
if __name__ == "__main__":
    print("Starting Ray Cluster and instantiating Actors...")
    extractor = GLiNERActor.remote()
    verifier = NLIActor.remote()
    firewall = SHACLFirewallActor.remote()

    reports = [
        "Study: Chest Radiograph. The cardiac silhouette is normal. There is a 8 mm nodule in the right upper lobe. Lungs are clear without pneumothorax.",
        "Study: CT Head. No acute intracranial hemorrhage or mass effect. The ventricles are normal in caliber. Mild cortical atrophy is noted.",
        "Study: MRI Lumbar Spine. L4-L5 demonstrates a 3 mm broad-based posterior disc bulge indenting the thecal sac. No severe central canal stenosis.",
        "Study: US Abdomen. The liver is echogenic, suggestive of hepatic steatosis. The gallbladder is unremarkable without gallstones.",
        "Study: CT Chest PE. No pulmonary embolism identified. A 1.2 cm left adrenal adenoma is incidentally noted. Heart size is normal.",
        "Study: X-Ray Knee. No acute fracture or dislocation. There is moderate medial compartment joint space narrowing with marginal osteophytes.",
        "Study: US Thyroid. A 2 cm solid hypoechoic nodule is present in the right lobe. The left lobe is unremarkable. No suspicious lymphadenopathy.",
        "Study: CT Pelvis. Acute appendicitis with a 1.1 cm dilated appendix and periappendiceal fat stranding. No free air or fluid."
    ]

    start_time = time.time()
    print(f"Processing {len(reports)} reports concurrently via Ray...")

    extraction_futures = [extractor.extract.remote(i, txt) for i, txt in enumerate(reports)]
    verification_futures = [verifier.verify.remote(ext) for ext in extraction_futures]
    validation_futures = [firewall.compile_and_validate.remote(ver) for ver in verification_futures]

    final_results = ray.get(validation_futures)
        
    print(f"\n--- HYBRID BATCH COMPLETE IN {time.time() - start_time:.2f} SECONDS ---")
    
    # Sort results by report_id to keep the output organized
    final_results = sorted(final_results, key=lambda x: x["report_id"])
    
    for res in final_results:
        idx = res['report_id']
        print(f"\n--- REPORT {idx} ---")
        print(f"RAW TEXT : {reports[idx]}")
        print(f"STATUS   : SHACL Valid: {res['valid']} -> Extracted {len(res['claims'])} Facts")
    
    print("\n==================================================")
    print("SAMPLE COMPILED JSON (REPORT 7)")
    print(f"RAW TEXT: {reports[7]}")
    print("==================================================")
    print(json.dumps(final_results[7]["claims"], indent=2))
    
    ray.shutdown()