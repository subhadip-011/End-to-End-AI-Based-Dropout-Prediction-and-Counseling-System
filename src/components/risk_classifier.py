import os
import sys
import dill
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException


# ---------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# ---------------------------------------------------------------
# RISK THRESHOLDS — change these values to tune sensitivity
#
# WHY these thresholds?
# - Below 0.35  → model is fairly confident student is safe
# - 0.35 - 0.65 → uncertain zone — needs monitoring
# - Above 0.65  → model strongly predicts dropout
#
# In real deployment, thresholds are tuned based on:
# - How many counselors are available (capacity)
# - Cost of missing a real dropout (false negative cost)
# ---------------------------------------------------------------
LOW_RISK_MAX    = 0.35   # 0.00 – 0.35 → Low Risk
MEDIUM_RISK_MAX = 0.65   # 0.35 – 0.65 → Medium Risk
                         # 0.65 – 1.00 → High Risk


# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
@dataclass
class RiskClassifierConfig:
    model_path:       str = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    preprocessor_path: str = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    output_path:      str = os.path.join(PROJECT_ROOT, "artifacts", "risk_scores.csv")


# ---------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------
class RiskClassifier:
    def __init__(self):
        self.config = RiskClassifierConfig()
        self.model        = self._load_object(self.config.model_path)
        self.preprocessor = self._load_object(self.config.preprocessor_path)

    def _load_object(self, path):
        """Load a saved .pkl object (model or preprocessor)."""
        with open(path, "rb") as f:
            obj = dill.load(f)
        logger.info(f"Loaded: {path}")
        return obj

    def _get_dropout_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw input and returns dropout probability
        for each student using the trained model.

        predict_proba() returns [[prob_graduate, prob_dropout], ...]
        We take [:, 1] — the dropout probability column only.
        """
        # Drop Target column if present (enrolled students still have it)
        if "Target" in df.columns:
            df = df.drop(columns=["Target"])

        transformed = self.preprocessor.transform(df)
        probabilities = self.model.predict_proba(transformed)[:, 1]
        return probabilities

    def _assign_risk_level(self, probability: float) -> str:
        """
        Converts a single dropout probability into a risk label.

        This is the core logic of the Risk Classification system.
        Simple, transparent, and easy to explain to non-technical
        stakeholders like school administrators.
        """
        if probability < LOW_RISK_MAX:
            return "Low Risk"
        elif probability < MEDIUM_RISK_MAX:
            return "Medium Risk"
        else:
            return "High Risk"

    def _get_risk_emoji(self, risk_level: str) -> str:
        """Visual indicator for dashboards."""
        mapping = {
            "Low Risk"    : "🟢",
            "Medium Risk" : "🟡",
            "High Risk"   : "🔴",
        }
        return mapping.get(risk_level, "⚪")

    def _get_counseling_urgency(self, risk_level: str) -> str:
        """
        Maps risk level to counseling action.
        This feeds directly into the Counseling Recommender (Step 10).
        """
        mapping = {
            "Low Risk"    : "No immediate action needed. Monitor monthly.",
            "Medium Risk" : "Schedule check-in within 2 weeks.",
            "High Risk"   : "Immediate counseling session required!",
        }
        return mapping.get(risk_level, "Unknown")

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method — takes a DataFrame of students and returns
        the same DataFrame enriched with:
        - dropout_probability
        - risk_level
        - risk_emoji
        - counseling_urgency

        Works for ANY student data:
        - Enrolled students (Early Warning System)
        - Single new student (Prediction Pipeline)
        - Batch of students (Teacher Dashboard)
        """
        logger.info("========== Risk Classification Started ==========")
        try:
            logger.info(f"Classifying {len(df)} students...")

            # Get dropout probabilities from model
            probabilities = self._get_dropout_probability(df.copy())

            # Build result DataFrame
            result_df = df.copy()
            result_df["dropout_probability"] = np.round(probabilities, 4)
            result_df["risk_level"]          = result_df["dropout_probability"].apply(
                self._assign_risk_level
            )
            result_df["risk_emoji"]          = result_df["risk_level"].apply(
                self._get_risk_emoji
            )
            result_df["counseling_urgency"]  = result_df["risk_level"].apply(
                self._get_counseling_urgency
            )

            # Log risk distribution
            risk_counts = result_df["risk_level"].value_counts().to_dict()
            logger.info(f"Risk Distribution: {risk_counts}")

            logger.info("========== Risk Classification Completed ==========")
            return result_df

        except Exception as e:
            raise CustomException(e, sys)

    def classify_single(self, student_dict: dict) -> dict:
        """
        Classifies a single student given as a dictionary.
        Used by the Student Dashboard and Flask API.

        Example input:
        {
            "Age at enrollment": 20,
            "Curricular units 1st sem (approved)": 5,
            ...
        }
        """
        logger.info("Classifying single student...")
        try:
            df = pd.DataFrame([student_dict])
            result = self.classify(df)

            return {
                "dropout_probability" : result["dropout_probability"].iloc[0],
                "risk_level"          : result["risk_level"].iloc[0],
                "risk_emoji"          : result["risk_emoji"].iloc[0],
                "counseling_urgency"  : result["counseling_urgency"].iloc[0],
            }

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------------------------------------------------
# RUN DIRECTLY TO TEST
# python src/components/risk_classifier.py
# ---------------------------------------------------------------
if __name__ == "__main__":

    # Load enrolled students — these are the ones we predict on
    enrolled_path = os.path.join(PROJECT_ROOT, "artifacts", "enrolled_students.csv")
    enrolled_df   = pd.read_csv(enrolled_path)

    logger.info(f"Loaded {len(enrolled_df)} enrolled students")

    # Classify all enrolled students
    classifier = RiskClassifier()
    result_df  = classifier.classify(enrolled_df)

    # Save results
    output_path = os.path.join(PROJECT_ROOT, "artifacts", "risk_scores.csv")
    result_df.to_csv(output_path, index=False)
    logger.info(f"Risk scores saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("       RISK CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total Enrolled Students : {len(result_df)}")
    print()

    for level in ["High Risk", "Medium Risk", "Low Risk"]:
        count  = (result_df["risk_level"] == level).sum()
        emoji  = result_df[result_df["risk_level"] == level]["risk_emoji"].iloc[0]
        pct    = round(count / len(result_df) * 100, 1)
        print(f"{emoji} {level:<15} : {count} students ({pct}%)")

    print("\n--- Sample High Risk Students ---")
    high_risk = result_df[result_df["risk_level"] == "High Risk"][
        ["dropout_probability", "risk_level", "counseling_urgency"]
    ].head(5)
    print(high_risk.to_string(index=False))