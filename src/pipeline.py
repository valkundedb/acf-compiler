import ray
import time
import json
from src.actors.gliner_actor import GLiNERActor
from src.actors.regex_actor import RegexActor
from src.actors.nli_actor import NLIActor
from src.actors.normalizer_actor import OntologyNormalizerActor
from src.actors.compiler_actor import ACFCompilerActor

def run_pipeline():
    ray.init(ignore_reinit_error=True)

    print("Instantiating distributed Ray actors...")
    semantic_extractor = GLiNERActor.remote()
    deterministic_extractor = RegexActor.remote()
    verifier = NLIActor.remote()
    normalizer = OntologyNormalizerActor.remote()
    compiler = ACFCompilerActor.remote()

    reports = [
        "Study: Chest Radiograph. The cardiac silhouette is normal. There is a 8 mm nodule in the right upper lobe. Lungs are clear without pneumothorax.",
        "Study: CT Head. No acute intracranial hemorrhage or mass effect. The ventricles are normal in caliber. Mild cortical atrophy is noted.",
        "Study: MRI Lumbar Spine. L4-L5 demonstrates a 3 mm broad-based posterior disc bulge indenting the thecal sac. No severe central canal stenosis.",
        "Study: US Abdomen. The liver is echogenic, suggestive of hepatic steatosis. The gallbladder is unremarkable without gallstones.",
        "Study: CT Chest PE. No pulmonary embolism identified. A 1.2 cm left adrenal adenoma is incidentally noted. Heart size is normal.",
        "Study: X-Ray Knee. No acute fracture or dislocation. There is moderate medial compartment joint space narrowing with marginal osteophytes.",
        "Study: US Thyroid. A 2 cm solid hypoechoic nodule is present in the right lobe. The left lobe is unremarkable. No suspicious lymphadenopathy.",
        "Study: CT Pelvis. Acute appendicitis with a 1.1 cm dilated appendix and periappendiceal fat stranding. No free air or fluid."
    ]

    start_time = time.time()
    print(f"Processing {len(reports)} reports concurrently via Ray DAG...")

    validation_futures = []
    for i, text in enumerate(reports):
        sem_future = semantic_extractor.extract.remote(i, text)
        det_future = deterministic_extractor.extract.remote(i, text)
        ver_future = verifier.verify.remote(sem_future)
        norm_future = normalizer.normalize.remote(ver_future)
        val_future = compiler.compile_and_validate.remote(norm_future, det_future)
        validation_futures.append(val_future)

    final_results = ray.get(validation_futures)
    
    print(f"\n--- HYBRID BATCH COMPLETE IN {time.time() - start_time:.2f} SECONDS ---")
    final_results = sorted(final_results, key=lambda x: x["report_id"])
    
    for res in final_results:
        idx = res['report_id']
        print(f"\n--- REPORT {idx} ---")
        print(f"RAW TEXT : {reports[idx]}")
        print(f"STATUS   : SHACL Valid: {res['valid']} -> Extracted {len(res['claims'])} Facts")
    
    print("\n==================================================")
    print("SAMPLE COMPILED JSON (REPORT 7 - WITH UMLS CUI MAPPING)")
    print("==================================================")
    print(json.dumps(final_results[7]["claims"], indent=2))
    
    ray.shutdown()

if __name__ == "__main__":
    run_pipeline()