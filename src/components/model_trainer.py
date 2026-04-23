import os
import sys
import numpy as np
import dill

from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

from src.logger import logger
from src.exception import CustomException


# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    report_path: str = os.path.join("artifacts", "model_report.txt")


# ---------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def _get_models(self):
        """
        Define all candidate models in one dictionary.

        WHY these 4 models?
        - LogisticRegression : Simple baseline — if complex models
                               don't beat this, something is wrong
        - RandomForest       : Strong ensemble, handles non-linearity
        - XGBoost            : Industry standard for tabular data
        - MLPClassifier      : Neural Network — adds DL component
        """
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1           # use all CPU cores
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            ),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(64, 32),   # 2 hidden layers
                activation="relu",
                max_iter=300,
                random_state=42
            ),
        }
        return models

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """
        Trains a model and returns key metrics.

        WHY these metrics?
        - Accuracy  : Overall correctness
        - F1 Score  : Balances precision & recall — important for
                      imbalanced data (more Graduates than Dropouts)
        - ROC-AUC   : How well model separates classes — best single
                      metric for binary classification
        """
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # dropout probability

        metrics = {
            "accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "f1_score" : round(f1_score(y_test, y_pred), 4),
            "roc_auc"  : round(roc_auc_score(y_test, y_pred_prob), 4),
        }
        return metrics

    def _save_object(self, obj, path):
        with open(path, "wb") as f:
            dill.dump(obj, f)
        logger.info(f"Object saved: {path}")

    def _save_report(self, report: dict):
        """
        Saves model comparison report as a text file.
        Useful for documentation and GitHub README.
        """
        with open(self.config.report_path, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("     MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            for model_name, metrics in report.items():
                f.write(f"Model     : {model_name}\n")
                f.write(f"Accuracy  : {metrics['accuracy']}\n")
                f.write(f"F1 Score  : {metrics['f1_score']}\n")
                f.write(f"ROC-AUC   : {metrics['roc_auc']}\n")
                f.write("-" * 40 + "\n")
        logger.info(f"Report saved: {self.config.report_path}")

    def initiate_model_training(self, train_arr, test_arr):
        """
        Main method:
        1. Split arrays into X, y
        2. Train all models and compare
        3. Select best model by ROC-AUC
        4. Save best model as model.pkl
        5. Return best model name and score
        """
        logger.info(" Model Training Started ")
        try:
            # Split features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

            # Train and evaluate all models
            models  = self._get_models()
            report  = {}
            trained = {}   # store trained model objects

            logger.info("Training and evaluating all models...")
            logger.info("-" * 40)

            for model_name, model in models.items():
                logger.info(f"Training: {model_name}")
                metrics = self._evaluate_model(
                    model, X_train, y_train, X_test, y_test
                )
                report[model_name]  = metrics
                trained[model_name] = model

                logger.info(
                    f"  Accuracy: {metrics['accuracy']} | "
                    f"F1: {metrics['f1_score']} | "
                    f"ROC-AUC: {metrics['roc_auc']}"
                )
                logger.info("-" * 40)

            # Select best model by ROC-AUC score
            best_model_name = max(
                report, key=lambda name: report[name]["roc_auc"]
            )
            best_model        = trained[best_model_name]
            best_score        = report[best_model_name]["roc_auc"]

            logger.info(f"Best Model : {best_model_name}")
            logger.info(f"Best ROC-AUC: {best_score}")

            # Minimum threshold — if best model is still bad, raise error
            if best_score < 0.70:
                raise CustomException(
                    "No model crossed ROC-AUC threshold of 0.70. "
                    "Check data quality.", sys
                )

            # Save best model
            self._save_object(best_model, self.config.model_path)

            # Save comparison report
            self._save_report(report)

            logger.info(" Model Training Completed ")
            return best_model_name, best_score, report

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------------------------------------------------
# RUN DIRECTLY TO TEST
# python src/components/model_trainer.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Step 1: Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion("notebook/data/student_data.csv")

    # Step 2: Transformation
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # Step 3: Model Training
    trainer = ModelTrainer()
    best_model_name, best_score, report = trainer.initiate_model_training(
        train_arr, test_arr
    )

    #print("\n" + "=" * 50)
    
    print(" FINAL MODEL COMPARISON REPORT ")
    print("=" * 50)
    for name, metrics in report.items():
        print(f"\n{name}")
        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  F1 Score : {metrics['f1_score']}")
        print(f"  ROC-AUC  : {metrics['roc_auc']}")

    print("\n" + "=" * 50)
    print(f"  Best Model : {best_model_name}")
    print(f"  ROC-AUC    : {best_score}")
    #print("=" * 50)