import os
import sys
import numpy as np
import pandas as pd

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_text_preprocessed.csv")

data = pd.read_csv(file_path)
data = data.dropna().drop_duplicates()

X = data["cleaned_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

input_example = np.array(X_test[0:5])
max_features = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
ngram_range = (1, int(sys.argv[2])) if len(sys.argv) > 2 else (1, 1)
c = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

with mlflow.start_run():
    svc = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("svc", LinearSVC(C=c))
    ]
    )
    svc.fit(X_train, y_train)

    mlflow.sklearn.log_model(svc,"model", input_example=input_example)

    y_pred = svc.predict(X_test)

    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
