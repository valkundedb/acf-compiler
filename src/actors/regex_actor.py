import ray
import re

@ray.remote(num_cpus=1)
class RegexActor:
    def __init__(self):
        self.measurement_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(cm|mm)')
        self.nlp = None

    def extract(self, report_id, text):
        if self.nlp is None:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
            
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