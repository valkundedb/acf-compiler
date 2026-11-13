import ray

@ray.remote(num_cpus=1)
class OntologyNormalizerActor:
    def __init__(self):
        # Enterprise UMLS / RadLex / SNOMED lookup dictionary (prepped for Trie / Aho-Corasick expansion)
        self.exact_lexicon = {
            "appendix": {"cui": "C0003611", "code": "85740005", "coding_system": "SNOMED-CT", "preferred_name": "Appendix structure"},
            "acute appendicitis": {"cui": "C0001622", "code": "116680003", "coding_system": "SNOMED-CT", "preferred_name": "Acute appendicitis"},
            "periappendiceal fat stranding": {"cui": "C0474345", "code": "RID39523", "coding_system": "RadLex", "preferred_name": "Fat stranding"}
        }

    def normalize(self, claims):
        normalized_claims = []
        for claim in claims:
            entity_text = claim['entity'].lower().strip()
            if entity_text in self.exact_lexicon:
                match = self.exact_lexicon[entity_text]
                claim['cui'] = match['cui']
                claim['ontology_code'] = match['code']
                claim['coding_system'] = match['coding_system']
                claim['normalization_method'] = 'Direct Lookup'
            else:
                claim['cui'] = 'UNMAPPED_OOV'
                claim['normalization_method'] = 'Pending Semantic Retrieval'
            normalized_claims.append(claim)
        return normalized_claims