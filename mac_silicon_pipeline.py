import spacy
import torch
from gliner import GLiNER
from transformers import pipeline
import json

# 1. Hardware Acceleration (Apple Silicon Metal GPU)
device_name = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)
print(f"--- Booting ML Engine on: {device_name.upper()} ---\n")

# 2. Load the Perception Engine (GLiNER)
# GLiNER performs zero-shot NER without hardcoded dictionaries.
gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(device)
labels_to_find = ["clinical finding", "anatomy", "disease"]

# 3. Load the Verifier Engine (NLI Cross-Encoder)
# Replaces the SVM. Treats assertion as a logical entailment problem.
nli_classifier = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=device
)
assertion_labels = ["Present", "Absent", "Normal"]

# 4. Ingest and Segment
nlp = spacy.load("en_core_web_sm")
raw_report = """Study: Chest Radiograph

The cardiac silhouette is normal in size and contour. The mediastinal contours are within normal limits. There is a 8 mm well-circumscribed nodule in the right upper lobe. The remainder of the lungs are clear without consolidation, pneumothorax, or pleural effusion. The pulmonary vasculature appears normal. No acute bony abnormalities are identified. The visualized upper abdomen is unremarkable."""

doc = nlp(raw_report)
extracted_claims = []

for sent in doc.sents:
    sentence_text = sent.text.strip()
    if not sentence_text or "Study:" in sentence_text:
        continue
        
    # Step A: Deep Learning Entity Extraction (GLiNER)
    entities = gliner_model.predict_entities(sentence_text, labels_to_find, threshold=0.4)
    
    for ent in entities:
        entity_text = ent["text"]
        
        # Step B: Natural Language Inference (NLI)
        # We ask the NLI model: Based on the sentence, what is the status of the entity?
        hypothesis_template = f"The {entity_text} is {{}}."
        
        nli_result = nli_classifier(
            sentence_text,
            candidate_labels=assertion_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )
        
        # The top predicted label (Present, Absent, Normal)
        top_assertion = nli_result['labels'][0]
        confidence = nli_result['scores'][0]
        
        extracted_claims.append({
            "entity": entity_text,
            "type": ent["label"],
            "assertion": top_assertion,
            "confidence": round(confidence, 3),
            "source_sentence": sentence_text
        })

print("--- EXTRACTED ATOMIC CLINICAL FACTS (DEEP LEARNING) ---")
print(json.dumps(extracted_claims, indent=2))