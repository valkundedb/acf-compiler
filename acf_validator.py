import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
from pyshacl import validate

# 1. Define Namespaces
RAD = Namespace("http://radlex.org/")
SH = Namespace("http://www.w3.org/ns/shacl#")
EX = Namespace("http://hospital.org/clinical/")

# 2. The Deterministic SHACL Rules
# Rule: A Validated ACF MUST contain an Assertion, Hedging, and Laterality.
# If the SVM missed one, the compiler throws an error.
shacl_rules = """
@prefix rad: <http://radlex.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://hospital.org/clinical/> .

ex:AtomicFactShape a sh:NodeShape ;
    sh:targetClass ex:AtomicClinicalFact ;
    sh:property [
        sh:path rad:hasAssertion ;
        sh:minCount 1 ;
        sh:message "COMPILER_ERROR: Missing mandatory Assertion facet." ;
    ] ;
    sh:property [
        sh:path rad:hasLaterality ;
        sh:minCount 1 ;
        sh:message "COMPILER_ERROR: Missing mandatory Laterality facet." ;
    ] ;
    sh:property [
        sh:path rad:hasHedging ;
        sh:minCount 1 ;
        sh:message "COMPILER_ERROR: Missing mandatory Hedging facet." ;
    ] .
"""

def validate_acf(svm_payload):
    print(f"--- Receiving SVM Payload ---")
    print(json.dumps(svm_payload, indent=2))
    
    # Initialize the data graph
    data_graph = Graph()
    
    # Create a unique URI for this finding instance
    finding_instance = EX["fact_001"]
    
    # Construct the Graph from the SVM Output
    data_graph.add((finding_instance, RDF.type, EX.AtomicClinicalFact))
    data_graph.add((finding_instance, RAD.hasFinding, Literal(svm_payload["finding"])))
    data_graph.add((finding_instance, RAD.hasAssertion, Literal(svm_payload["assertion"])))
    data_graph.add((finding_instance, RAD.hasHedging, Literal(svm_payload["hedging"])))
    data_graph.add((finding_instance, RAD.hasLaterality, Literal(svm_payload["laterality"])))
    
    # Load SHACL graph
    shacl_graph = Graph().parse(data=shacl_rules, format="turtle")
    
    # Run the Deterministic Validation
    print("\n--- Running SHACL Firewall ---")
    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shacl_graph,
        inference='rdfs',
        abort_on_first=False,
        meta_shacl=False,
        debug=False
    )
    
    if conforms:
        print("STATUS: VALIDATED [200 OK]")
        print("The clinical claim is structurally sound and ready for MILP Optimizer routing.")
    else:
        print("STATUS: INVALID_CLAIM [400 Bad Request]")
        error_msg = results_text.split("Message: ")[-1].split("\n")[0] if "Message: " in results_text else "Structural Error"
        print(f"Reason: {error_msg}")

# --- Execute with the exact output from your SVM run ---
svm_output = {
    "finding": "Calculus",
    "assertion": "Present",
    "hedging": "Possible",
    "severity": "Mild",
    "laterality": "Right"
}

validate_acf(svm_output)