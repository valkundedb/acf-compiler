import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV

# 1. Expanded Synthetic Training Data
data = pd.DataFrame({
    'text': [
        "mild right hydronephrosis is present",
        "no evidence of left-sided calculus",
        "suspicious for severe bilateral pneumonia",
        "history of calculus, currently absent",
        "moderate left hydronephrosis noted",
        "cannot exclude mild right calculus",
        "severe right calculus is present",
        "cannot exclude left pneumonia",
        "mild bilateral calculus noted"
    ],
    'claim_length': [6, 6, 5, 5, 4, 5, 5, 4, 4],
    'assertion': ['Present', 'Absent', 'Present', 'Absent', 'Present', 'Present', 'Present', 'Present', 'Present'],
    'hedging':   ['Certain', 'Certain', 'Possible', 'Certain', 'Certain', 'Possible', 'Certain', 'Possible', 'Certain'],
    'severity':  ['Mild', 'None', 'Severe', 'None', 'Moderate', 'Mild', 'Severe', 'None', 'Mild'],
    'laterality':['Right', 'Left', 'Bilateral', 'None', 'Left', 'Right', 'Right', 'Left', 'Bilateral']
})

# FIX: Artificially duplicate the dataset 5 times to satisfy 5-fold CV limits
data = pd.concat([data] * 5, ignore_index=True)

X = data[['text', 'claim_length']]
Y = data[['assertion', 'hedging', 'severity', 'laterality']]

# 2. Feature Extraction Pipeline
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3)))
])

def extract_length(df):
    return df[['claim_length']]

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'text'),
        ('length', FunctionTransformer(extract_length), ['claim_length'])
    ])

# 3. The Multi-Output SVM 
base_svm = SVC(kernel='linear', C=1.0)
calibrated_svm = CalibratedClassifierCV(base_svm, ensemble=False)

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(calibrated_svm))
])

# Train the model
svm_pipeline.fit(X, Y)

# 4. Testing the Engine
test_claim = pd.DataFrame({
    'text': ["cannot exclude mild right calculus"],
    'claim_length': [5]
})

predictions = svm_pipeline.predict(test_claim)
probabilities = svm_pipeline.predict_proba(test_claim)

print(f"Input Text: '{test_claim['text'].iloc[0]}'")
print("-" * 40)
facets = ['Assertion', 'Hedging', 'Severity', 'Laterality']
for idx, facet in enumerate(facets):
    pred_class = predictions[0][idx]
    class_idx = list(svm_pipeline.named_steps['classifier'].estimators_[idx].classes_).index(pred_class)
    conf = probabilities[idx][0][class_idx]
    
    print(f"{facet}: {pred_class} (Confidence: {conf:.2f})")