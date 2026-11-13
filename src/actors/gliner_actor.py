import ray

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