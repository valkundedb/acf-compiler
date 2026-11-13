# Atomic Clinical Fact (ACF) Compiler: Architecture & Design Specification

## 1. Executive Summary
The Atomic Clinical Fact (ACF) Compiler is a high-throughput, deterministic, polyglot data pipeline designed to transform unstructured radiology reports into validated Labeled Property Graphs (LPG). By decoupling deep learning perception from deterministic symbolic execution and enforcing strict ontological constraints, the system ensures atomic granularity required for longitudinal delta tracking across patient priors.

## 2. Distributed Execution Topology (Ray Cluster)
The compiler leverages an actor-based distributed Directed Acyclic Graph (DAG) running on a Ray cluster:
- **GLiNERActor (GPU Pool):** Executes span-based semantic boundary prediction using `gliner-community/gliner_medium-v2.5` to extract core clinical entities (`Disease`, `Anatomy`, `Clinical finding`, `Modifier`, `Substance`).
- **RegexActor (CPU Pool):** Executes deterministic pattern matching for dimensional measurements (`cm`, `mm`) with exact character offsets.
- **NLIActor (GPU Pool):** Evaluates assertion states (`Present`, `Absent`, `Normal`) using cross-encoder NLI models (`cross-encoder/nli-deberta-v3-small`).
- **OntologyNormalizerActor (CPU Pool):** Maps extracted text to official UMLS, RadLex, and SNOMED CT identifiers using a cascading resolution strategy.
- **ACFCompilerActor (CPU Pool):** Performs ScispaCy-based syntactic dependency parsing, tree BFS edge resolution, proximity tie-breaking, and SHACL constraint validation.

## 3. Hybrid Cascaded Normalization (UMLS Integration)
To leverage your enterprise UMLS license efficiently without incurring excessive inference latency, normalization occurs across three deterministic tiers:
1. **Tier 1 ($O(1)$ Trie Lookup):** In-memory prefix lookup across cached canonical terms, preferred names, and high-frequency clinical synonyms mapped to UMLS Concept Unique Identifiers (CUIs).
2. **Tier 2 (Dense Semantic Vector Similarity):** Fallback embedding comparison using biomedical encoders (`pritamdeka/SBERT-pubmedBERT-base-minilm`) against pre-indexed vector spaces for out-of-vocabulary (OOV) expressions.
3. **Tier 3 (Contextual Graph Validation):** Resolution of ambiguous terms utilizing existing relationship edges (`LOCATED_AT`, `MODIFIES`) within the local LPG context.

## 4. SHACL Firewall & Graph Validation
The compiled facts are structured into an RDF/OWL data graph and validated against pre-compiled SHACL shapes. The firewall guarantees:
- Mandatory property counts (`sh:minCount`).
- Controlled vocabulary assertion boundaries (`sh:in ("Present" "Absent" "Normal")`).
- Strict topological typing for entity-to-modifier relationships.