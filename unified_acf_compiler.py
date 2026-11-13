import ray
import json
import time
import re

ray.init(ignore_reinit_error=True)

# --- 1. THE SEMANTIC ACTOR (GPU) ---
@ray.remote(num_cpus=1)
class GLiNERActor:
    def __init__(self):
        self.labels = ["Disease", "Anatomy", "Clinical finding", "Modifier", "Substance"]
        self.model = None
        self.nlp = None

    def _initialize_models(self):
        if self.model is None:
            import torch
            import spacy
            from gliner import GLiNER
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = GLiNER.from_pretrained("gliner-community/gliner_medium-v2.5").to(self.device)
            self.nlp = spacy.load("en_core_sci_sm")

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
                    "source_sentence": sentence_text,
                    "extraction_method": "Deep Learning (GLiNER)",
                    "start": ent["start"],
                    "end": ent["end"]
                })
        return results

# --- 2. THE DETERMINISTIC ACTOR (CPU) ---
@ray.remote(num_cpus=1)
class RegexActor:
    def __init__(self):
        self.measurement_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(cm|mm)')
        self.nlp = None

    def extract(self, report_id, text):
        if self.nlp is None:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            
        doc = self.nlp(text)
        results = []
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            for match in self.measurement_pattern.finditer(sentence_text):
                results.append({
                    "report_id": report_id,
                    "entity": match.group(0),
                    "type": "Measurement",
                    "source_sentence": sentence_text,
                    "extraction_method": "Regex (CPU)",
                    "assertion": "Present", 
                    "confidence": 1.0,
                    "start": match.start(),
                    "end": match.end()
                })
        return results

# --- 3. THE VERIFIER ACTOR (GPU) ---
@ray.remote(num_cpus=1)
class NLIActor:
    def __init__(self):
        self.assertion_labels = ["Present", "Absent", "Normal"]
        self.classifier = None

    def _initialize_models(self):
        if self.classifier is None:
            import torch
            from transformers import pipeline
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=device
            )

    def verify(self, semantic_entities):
        self._initialize_models()
        verified_claims = []
        for item in semantic_entities:
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

# --- 4. THE COMPILER & FIREWALL ACTOR (CPU) ---
@ray.remote(num_cpus=1)
class ACFCompilerActor:
    def __init__(self):
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
        self.nlp = None

    def _get_syntactic_distance(self, doc, span1_start, span1_end, span2_start, span2_end):
        """Calculates shortest path through the grammatical dependency tree using BFS"""
        # Robust token alignment ignoring whitespace and boundary artifacts
        tokens1 = [t for t in doc if t.idx < span1_end and (t.idx + len(t.text)) > span1_start and not t.is_space and not t.is_punct]
        tokens2 = [t for t in doc if t.idx < span2_end and (t.idx + len(t.text)) > span2_start and not t.is_space and not t.is_punct]
        
        if not tokens1 or not tokens2:
            return float('inf')

        # Build an undirected adjacency list from the dependency tree
        adj = {t.i: [] for t in doc}
        for t in doc:
            if t.head.i != t.i:
                adj[t.i].append(t.head.i)
                adj[t.head.i].append(t.i)

        # BFS from tokens1 to tokens2
        from collections import deque
        queue = deque([(t.i, 0) for t in tokens1])
        visited = set(t.i for t in tokens1)
        target_indices = set(t.i for t in tokens2)

        while queue:
            curr, dist = queue.popleft()
            if curr in target_indices:
                return dist
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def compile_and_validate(self, verified_semantic, deterministic):
        from rdflib import Graph, Literal
        from rdflib.namespace import RDF
        from pyshacl import validate
        
        if self.nlp is None:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
            
        all_claims = verified_semantic + deterministic
        if not all_claims:
            return {"report_id": None, "valid": False, "claims": []}
            
        report_id = all_claims[0]["report_id"]

        # --- SYNTACTIC EDGE RESOLUTION (FUSION) ---
        sentences = set(c['source_sentence'] for c in all_claims)
        
        for sentence in sentences:
            doc = self.nlp(sentence)
            nodes_in_sent = [c for c in all_claims if c['source_sentence'] == sentence]
            measurements = [n for n in nodes_in_sent if n['type'] == 'Measurement']
            targets = [n for n in nodes_in_sent if n['type'] in ['Anatomy', 'Clinical finding', 'Disease']]
            
            for meas in measurements:
                closest_target = None
                min_distance = float('inf')
                min_char_dist = float('inf')
                
                for tgt in targets:
                    # 1. Primary Filter: Trace the ScispaCy grammatical tree
                    dist = self._get_syntactic_distance(doc, meas['start'], meas['end'], tgt['start'], tgt['end'])
                    
                    # 2. Secondary Filter: Raw physical character distance
                    char_dist = min(abs(meas['start'] - tgt['end']), abs(tgt['start'] - meas['end']))
                    
                    # --- DEBUG LOGGING ---
                    if report_id == 7:
                        print(f"[DEBUG] Meas: '{meas['entity']}' | Target: '{tgt['entity']}' | Syn Dist: {dist} | Char Dist: {char_dist}")
                    
                    # Logic: Favor the shortest syntactic path
                    if dist < min_distance:
                        min_distance = dist
                        min_char_dist = char_dist
                        closest_target = tgt
                    
                    # TIE-BREAKER: If they are grammatical siblings (e.g., joined by "and")
                    elif dist == min_distance:
                        if char_dist < min_char_dist:
                            min_char_dist = char_dist
                            closest_target = tgt
                
                if closest_target:
                    meas['modifies_entity'] = closest_target['entity']
                    meas['modifies_type'] = closest_target['type']
                    meas['syntactic_distance'] = min_distance

        # --- SHACL GRAPH CONSTRUCTION ---
        data_graph = Graph()
        for idx, claim in enumerate(all_claims):
            fact_uri = self.EX[f"fact_{report_id}_{idx}"]
            data_graph.add((fact_uri, RDF.type, self.EX.ClinicalFact))
            data_graph.add((fact_uri, self.RAD.hasEntity, Literal(claim['entity'])))
            data_graph.add((fact_uri, self.RAD.hasAssertion, Literal(claim['assertion'])))
            
            if 'modifies_entity' in claim:
                data_graph.add((fact_uri, self.RAD.modifies, Literal(claim['modifies_entity'])))
        
        conforms, _, _ = validate(data_graph, shacl_graph=self.shacl_graph)
        return {"report_id": report_id, "valid": conforms, "claims": all_claims}
# --- 5. ORCHESTRATION SCRIPT ---
if __name__ == "__main__":
    print("Starting Ray Cluster and instantiating Actors...")
    semantic_extractor = GLiNERActor.remote()
    deterministic_extractor = RegexActor.remote()
    verifier = NLIActor.remote()
    compiler = ACFCompilerActor.remote()

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

    validation_futures = []
    for i, text in enumerate(reports):
        sem_future = semantic_extractor.extract.remote(i, text)
        det_future = deterministic_extractor.extract.remote(i, text)
        ver_future = verifier.verify.remote(sem_future)
        val_future = compiler.compile_and_validate.remote(ver_future, det_future)
        validation_futures.append(val_future)

    final_results = ray.get(validation_futures)
    
    print(f"\n--- HYBRID BATCH COMPLETE IN {time.time() - start_time:.2f} SECONDS ---")
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