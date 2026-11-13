import ray

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