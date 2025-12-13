import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)


class TextClassifierModel:
    """
    A unified machine learning manager that trains, tests, and evaluates
    three models for binary text classification:

        - Logistic Regression
        - Support Vector Machine (SVM)
        - Random Forest

    This class allows training, predictions, evaluation metrics, 
    and confusion matrix plotting in a structured, reusable form.
    """

    def __init__(self, scale_for_svm=None):
        """
        Initialize model objects.

        Parameters
        ----------
        scale_for_svm : sklearn scaler or None
            Optional scaler used only for SVM training/testing.
            Example: StandardScaler(), MinMaxScaler()
        """
        self.log_reg = LogisticRegression(max_iter=2000)
        self.svm = SVC(kernel="rbf", probability=True)
        self.rf = RandomForestClassifier(n_estimators=300, random_state=42)

        self.scaler = scale_for_svm  
        self.trained = False


    def train(self, X_train, y_train):
        """
        Train all models on pre-extracted features.

        Parameters
        ----------
        X_train : DataFrame or numpy array
            Training feature matrix.
        y_train : array or Series
            Training labels.
        """
        X_svm = self.scaler.transform(X_train) if self.scaler else X_train

        self.log_reg.fit(X_train, y_train)
        self.svm.fit(X_svm, y_train)
        self.rf.fit(X_train, y_train)

        self.trained = True
        print("\n Training Completed — All 3 Models Fit Successfully!\n")


    def evaluate(self, model, X_test, y_test):
        """
        Compute core evaluation metrics for a model.

        Parameters
        ----------
        model : sklearn estimator
            Model to evaluate.
        X_test : array-like
            Test dataset features.
        y_test : array-like
            True labels.

        Returns
        -------
        dict
            Dictionary containing accuracy, precision, recall, F1, AUC.
        """
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, pos_label="ai_generated"),
            "Recall": recall_score(y_test, preds, pos_label="ai_generated"),
            "F1-score": f1_score(y_test, preds, pos_label="ai_generated"),
            "ROC-AUC": roc_auc_score(y_test, probs) if probs is not None else None
        }

        print("\n Model Evaluation Metrics:")
        for k,v in metrics.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        return metrics


    def confusion_plot(self, model, X_test, y_test, title):
        """
        Generate a confusion matrix plot for a model.

        Parameters
        ----------
        model : sklearn estimator
            Trained classifier.
        X_test : feature matrix
        y_test : true labels
        title : str
            Title displayed above the plot.
        """
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds, labels=["human","ai_generated"])

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Human","AI"], yticklabels=["Human","AI"])
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


    def evaluate_all(self, X_test, y_test):
        """
        Run evaluation for all three models + confusion matrices.

        Parameters
        ----------
        X_test : feature matrix
        y_test : true labels

        Returns
        -------
        dict
            Dictionary containing metric results for all models.
        """
        if not self.trained:
            raise RuntimeError("Train the models first using .train()")

        X_svm = self.scaler.transform(X_test) if self.scaler else X_test
        
        print("\n================ MODEL PERFORMANCE SUMMARY ================\n")

        results = {
            "Logistic Regression": self.evaluate(self.log_reg, X_test, y_test),
            "SVM": self.evaluate(self.svm, X_svm, y_test),
            "Random Forest": self.evaluate(self.rf, X_test, y_test)
        }

        # Confusion Matrices
        self.confusion_plot(self.log_reg, X_test, y_test, "Logistic Regression CM")
        self.confusion_plot(self.svm, X_svm, y_test, "SVM CM")
        self.confusion_plot(self.rf, X_test, y_test, "Random Forest CM")

        return results
