import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import json
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
from pyshacl import validate

# 1. Initialize NLP & Namespaces
nlp = spacy.load("en_core_web_sm")
RAD = Namespace("http://radlex.org/")
EX = Namespace("http://hospital.org/clinical/")

# 2. The Clinical Dictionary
clinical_glossary = {
    "cardiac silhouette": RAD.Heart,
    "mediastinal contours": RAD.Mediastinum,
    "nodule": RAD.Nodule,
    "consolidation": RAD.Consolidation,
    "pneumothorax": RAD.Pneumothorax,
    "pleural effusion": RAD.PleuralEffusion,
    "pulmonary vasculature": RAD.PulmonaryVasculature,
    "bony abnormalities": RAD.Bone,
    "upper abdomen": RAD.Abdomen
}

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for term in clinical_glossary.keys():
    # FIX 1: Use nlp(term) instead of nlp.make_doc for perfect token alignment
    matcher.add("CLINICAL_ENTITY", [nlp(term)]) 

# 3. Train the Context SVM (The Proposer)
train_data = pd.DataFrame({
    'text': [
        "normal in size and contour",
        "within normal limits",
        "there is a well-circumscribed nodule",
        "clear without consolidation or effusion",
        "appears normal",
        "no acute abnormalities are identified",
        "is unremarkable",
        "severe consolidation noted"
    ],
    'claim_length': [6, 3, 5, 6, 2, 5, 2, 3],
    'assertion': ['Normal', 'Normal', 'Present', 'Absent', 'Normal', 'Absent', 'Normal', 'Present']
})

preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(ngram_range=(1, 3)), 'text'),
    ('length', FunctionTransformer(lambda df: df[['claim_length']]), ['claim_length'])
])
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', C=1.0))
])
svm_pipeline.fit(train_data[['text', 'claim_length']], train_data['assertion'])

# 4. Define the Multi-Axial SHACL Ontology
ontology_and_rules = """
@prefix rad: <http://radlex.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://hospital.org/clinical/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

rad:Nodule rdfs:subClassOf rad:Finding .
rad:Consolidation rdfs:subClassOf rad:Finding .
rad:Pneumothorax rdfs:subClassOf rad:Finding .
rad:PleuralEffusion rdfs:subClassOf rad:Finding .
rad:Heart rdfs:subClassOf rad:Anatomy .
rad:Bone rdfs:subClassOf rad:Anatomy .

ex:ClinicalFactShape a sh:NodeShape ;
    sh:targetClass ex:AtomicClinicalFact ;
    sh:property [
        sh:path rad:hasAssertion ;
        sh:minCount 1 ;
        sh:in ("Present" "Absent" "Normal") ;
        sh:message "COMPILER ERROR: Missing or invalid Assertion facet." ;
    ] .
"""

# 5. The End-to-End Orchestrator
def process_and_compile(raw_text):
    doc = nlp(raw_text)
    compiled_graph = Graph().parse(data=ontology_and_rules, format="turtle")
    shacl_graph = Graph().parse(data=ontology_and_rules, format="turtle")
    
    extracted_facts = []

    for sent_idx, sent in enumerate(doc.sents):
        # FIX 2: Create a fresh Doc for the sentence so token indices reset to 0
        sent_doc = nlp(sent.text) 
        matches = matcher(sent_doc)
        
        for match_id, start, end in matches:
            span = sent_doc[start:end]
            entity_text = span.text.lower()
            
            # Dictionary lookup now works perfectly
            ontology_node = clinical_glossary[entity_text] 
            
            tokens = sent.text.split()
            claim_length = len(tokens)
            
            df_input = pd.DataFrame({'text': [sent.text], 'claim_length': [claim_length]})
            assertion_pred = svm_pipeline.predict(df_input)[0]
            
            fact = {
                "entity_text": entity_text,
                "ontology_uri": ontology_node,
                "assertion": assertion_pred,
                "sentence": sent.text.strip()
            }
            extracted_facts.append(fact)
            
            fact_uri = EX[f"fact_{sent_idx}_{start}"]
            compiled_graph.add((fact_uri, RDF.type, EX.AtomicClinicalFact))
            compiled_graph.add((fact_uri, RAD.hasEntity, ontology_node))
            compiled_graph.add((fact_uri, RAD.hasAssertion, Literal(assertion_pred)))

    conforms, _, results_text = validate(compiled_graph, shacl_graph=shacl_graph, inference='rdfs')
    
    print(f"--- EXTRACTED & COMPILED ATOMIC CLINICAL FACTS (Total: {len(extracted_facts)}) ---")
    print(json.dumps([{
        "entity": f["entity_text"],
        "assertion": f["assertion"],
        "ontology_mapping": str(f["ontology_uri"])
    } for f in extracted_facts], indent=2))
    
    if conforms:
        print("\nSTATUS: VALIDATED [200 OK] - Entire graph satisfies clinical structural rules.")
    else:
        print(f"\nSTATUS: INVALID_CLAIM [400]\nReason: {results_text}")

# --- EXECUTE ---
raw_report = """Study: Chest Radiograph

The cardiac silhouette is normal in size and contour. The mediastinal contours are within normal limits. There is a 8 mm well-circumscribed nodule in the right upper lobe. The remainder of the lungs are clear without consolidation, pneumothorax, or pleural effusion. The pulmonary vasculature appears normal. No acute bony abnormalities are identified. The visualized upper abdomen is unremarkable."""

process_and_compile(raw_report)