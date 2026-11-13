import spacy
import re
import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
from pyshacl import validate

# 1. Initialize NLP & Graph Namespaces
nlp = spacy.load("en_core_web_sm")
RAD = Namespace("http://radlex.org/")
EX = Namespace("http://hospital.org/clinical/")

# 2. Train the Assertion SVM (Dynamic Feature Extraction)
# We train it to recognize that "without" or "no" means Absent.
train_data = pd.DataFrame({
    'text': [
        "there is a 8 mm nodule present",
        "lungs are clear without consolidation",
        "no acute bony abnormalities identified",
        "mild hydronephrosis noted",
        "without pneumothorax or pleural effusion"
    ],
    'claim_length': [7, 5, 5, 3, 5],
    'assertion': ['Present', 'Absent', 'Absent', 'Present', 'Absent']
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

# 3. Define the SHACL Ontology & Rules
# Rule: A Nodule MUST have a Size and a Location in the Respiratory System.
ontology_and_rules = """
@prefix rad: <http://radlex.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://hospital.org/clinical/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Base Ontology Mapping
rad:RightUpperLobe rdfs:subClassOf rad:RespiratorySystem .

# SHACL Firewall Rules
ex:NoduleShape a sh:NodeShape ;
    sh:targetClass rad:Nodule ;
    sh:property [
        sh:path rad:hasLocation ;
        sh:class rad:RespiratorySystem ;
        sh:minCount 1 ;
        sh:message "ONTOLOGY ERROR: Nodule location must be in the Respiratory System." ;
    ] ;
    sh:property [
        sh:path rad:hasSize ;
        sh:minCount 1 ;
        sh:message "COMPILER ERROR: Nodule claim is missing a mandatory size measurement." ;
    ] .
"""

# 4. The Extraction Engine (Simulating the Rust/Ray pipeline)
def process_radiology_report(raw_text):
    print(f"--- INGESTING RAW REPORT ---\n{raw_text}\n")
    
    doc = nlp(raw_text)
    extracted_facts = []

    # Step A: Sentence Boundary Detection
    for sent in doc.sents:
        text = sent.text.lower()
        
        # Step B: Entity Identification (NER)
        if "nodule" in text:
            print(f"-> Target Entity Found in: '{sent.text}'")
            
            # Step C: Dynamic Feature Calculation for SVM
            tokens = text.split()
            claim_length = len(tokens) # Dynamically calculated, not hardcoded!
            
            # Step D: SVM Prediction for Assertion
            df_input = pd.DataFrame({'text': [text], 'claim_length': [claim_length]})
            prediction = svm_pipeline.predict(df_input)[0]
            
            # Step E: Deterministic Facet Extraction (Regex for Size & Location)
            size_match = re.search(r'(\d+(\.\d+)?)\s*(mm|cm)', text)
            size_val = size_match.group(0) if size_match else None
            
            location_val = None
            if "right upper lobe" in text:
                location_val = RAD.RightUpperLobe
                
            # Construct the Proposed ACF JSON
            fact = {
                "entity": RAD.Nodule,
                "assertion": prediction,
                "size": size_val,
                "location": location_val
            }
            extracted_facts.append(fact)
            
    return extracted_facts

# 5. The Compilation & Validation Engine
# 5. The Compilation & Validation Engine
def compile_facts(facts):
    shacl_graph = Graph().parse(data=ontology_and_rules, format="turtle")
    
    for idx, fact in enumerate(facts):
        data_graph = Graph().parse(data=ontology_and_rules, format="turtle")
        finding_uri = EX[f"finding_{idx}"]
        
        # Write the ACF to the RDF Graph
        data_graph.add((finding_uri, RDF.type, fact["entity"]))
        
        if fact.get("size"):
            data_graph.add((finding_uri, RAD.hasSize, Literal(fact["size"])))
        
        # THE FIX: Create a specific instance of the anatomy for this patient
        if fact.get("location"):
            loc_instance_uri = EX[f"patient_anatomy_{idx}"]
            
            # 1. Define the new node as an instance of the specific RadLex class
            data_graph.add((loc_instance_uri, RDF.type, fact["location"]))
            
            # 2. Link the finding to this specific physical instance
            data_graph.add((finding_uri, RAD.hasLocation, loc_instance_uri))
            
        # Run SHACL Validation
        conforms, _, results_text = validate(data_graph, shacl_graph=shacl_graph, inference='rdfs')
        
        print("\n--- COMPILED ATOMIC CLINICAL FACT (ACF) ---")
        print(json.dumps({
            "entity": "Nodule",
            "assertion": fact["assertion"],
            "size": fact["size"],
            "location_class": str(fact["location"])
        }, indent=2))
        
        if conforms:
            print("STATUS: VALIDATED [200 OK]")
        else:
            error = results_text.split("Message: ")[-1].split("\n")[0]
            print(f"STATUS: INVALID_CLAIM [400]\nReason: {error}")
# --- EXECUTE WITH RAW REPORT ---
raw_report = """Study: Chest Radiograph

The cardiac silhouette is normal in size and contour. The mediastinal contours are within normal limits. There is a 8 mm well-circumscribed nodule in the right upper lobe. The remainder of the lungs are clear without consolidation, pneumothorax, or pleural effusion. The pulmonary vasculature appears normal. No acute bony abnormalities are identified. The visualized upper abdomen is unremarkable."""

proposed_facts = process_radiology_report(raw_report)
compile_facts(proposed_facts)