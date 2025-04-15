from zenml import step, pipeline
# A framework for building machine learning pipelines.
from typing_extensions import Annotated
#Provides additional typing features.
from sklearn.datasets import load_digits
#Contains datasets for testing and example purposes
from sklearn.model_selection import train_test_split
#Provides functions for splitting datasets.
from sklearn.svm import SVC
#Contains the Support Vector Machine (SVM) classifier.
from sklearn.base import ClassifierMixin
#: Base classes for scikit-learn estimators
import numpy as np
import pandas as pd
from typing import Tuple
#Provides support for type hints.

# Step 1: Import data
#ZML (Zero-shot Machine Learning) 
#pipelines are designed to leverage pre-trained models to/ 
#perform tasks without requiring task-specific training data./
@step
#Decorator from ZenML to define a step in the pipeline.
def importer() -> Tuple[
#importer function: Loads the digits dataset, reshapes it, and splits it into training and testing sets.
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """Load digits dataset and split into train/test sets."""
    digits = load_digits()
#load_digits: Loads the digits dataset from scikit-learn.
    data = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(
#train_test_split: Splits the data into training and testing sets.
        data, digits.target, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# Step 2: Train model
@step
def svc_trainer(
#svc_trainer function: Trains an SVM classifier using the training data.
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train an sklearn SVC Classifier."""
    model = SVC(gamma=0.001)
#svc-SVC: Support Vector Classifier from scikit-learn.
    model.fit(X_train, y_train)
#model.fit: Trains the model on the training data.
    return model

# Step 3: Evaluate model
@step
def evaluator(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: ClassifierMixin,
) -> float:
    """Calculate the test set accuracy of the model."""
    test_acc = model.score(x_test, y_test)
    print(f"Test accuracy: {test_acc}")
    return test_acc

# Pipeline to link all steps
@pipeline
def digits_pipeline():
    """Connects the steps of the ML workflow."""
    x_train, x_test, y_train, y_test = importer()
    model = svc_trainer(X_train=x_train, y_train=y_train)
    evaluator(x_test=x_test, y_test=y_test, model=model)

if __name__ == "__main__":
    # This runs the pipeline and returns a PipelineRunResponse
    run_info = digits_pipeline()

    # Optional: print the run ID or other metadata
    print(f"✅ Pipeline run completed. Run ID: {run_info.id}")