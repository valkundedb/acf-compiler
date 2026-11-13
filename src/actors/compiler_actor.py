import ray
from src.utils.syntactic_parser import get_syntactic_distance

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

    def compile_and_validate(self, normalized_claims, deterministic):
        from rdflib import Graph, Literal
        from rdflib.namespace import RDF
        from pyshacl import validate
        import spacy
        
        if self.nlp is None:
            self.nlp = spacy.load("en_core_sci_sm")
            
        all_claims = normalized_claims + deterministic
        if not all_claims:
            return {"report_id": None, "valid": False, "claims": []}
            
        report_id = all_claims[0]["report_id"]

        # --- SYNTACTIC EDGE RESOLUTION (FUSION & TIE-BREAKER) ---
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
                    dist = get_syntactic_distance(doc, meas['start'], meas['end'], tgt['start'], tgt['end'])
                    char_dist = min(abs(meas['start'] - tgt['end']), abs(tgt['start'] - meas['end']))
                    
                    if dist < min_distance:
                        min_distance = dist
                        min_char_dist = char_dist
                        closest_target = tgt
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