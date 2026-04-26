import os
import sys
import dill
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without display
import matplotlib.pyplot as plt

from dataclasses import dataclass
from src.logger import logger
from src.exception import CustomException


# ---------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


# ---------------------------------------------------------------
# WHAT IS SHAP?
#
# SHAP = SHapley Additive exPlanations
# It answers: "WHY did the model give this student a high risk score?"
#
# For each student, SHAP assigns a value to every feature:
#   Positive SHAP value → pushes toward Dropout (increases risk)
#   Negative SHAP value → pushes toward Graduate (decreases risk)
#
# Example output for a High Risk student:
#   "Low 2nd sem grades"     → +0.45  (biggest dropout driver)
#   "Tuition fees not paid"  → +0.32  (second biggest driver)
#   "Has scholarship"        → -0.18  (protective factor)
#
# This is what makes our system EXPLAINABLE — not a black box.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
@dataclass
class ExplainerConfig:
    model_path:        str = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    preprocessor_path: str = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    plots_dir:         str = os.path.join(PROJECT_ROOT, "artifacts", "xai_plots")


# ---------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------
class DropoutExplainer:
    def __init__(self):
        self.config       = ExplainerConfig()
        self.model        = self._load_object(self.config.model_path)
        self.preprocessor = self._load_object(self.config.preprocessor_path)
        self.explainer    = None
        self.feature_names = None
        os.makedirs(self.config.plots_dir, exist_ok=True)

    def _load_object(self, path):
        with open(path, "rb") as f:
            return dill.load(f)

    def _get_feature_names(self) -> list:
        """
        Extracts feature names from the ColumnTransformer preprocessor.
        Needed so SHAP plots show real column names, not just indices.
        """
        continuous_features = self.preprocessor\
            .transformers_[0][2]   # first transformer's feature list
        binary_features = self.preprocessor\
            .transformers_[1][2]   # second transformer's feature list
        return list(continuous_features) + list(binary_features)

    def build_explainer(self, X_train_df: pd.DataFrame):
        """
        Builds the SHAP TreeExplainer using training data.

        WHY TreeExplainer?
        Our model is Random Forest — a tree-based model.
        TreeExplainer is optimized specifically for tree models.
        It is EXACT (not approximate) and very fast.

        Must be called once after training before explaining anything.
        """
        logger.info("Building SHAP TreeExplainer...")

        # Get feature names from preprocessor
        self.feature_names = self._get_feature_names()

        # Drop target column if present
        if "Target" in X_train_df.columns:
            X_train_df = X_train_df.drop(columns=["Target"])

        # Transform training data
        X_transformed = self.preprocessor.transform(X_train_df)
        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=self.feature_names
        )

        # Build SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP TreeExplainer built successfully.")

        return X_transformed_df

    def explain_student(self, student_df: pd.DataFrame) -> dict:
        """
        Explains WHY a single student is at risk.

        Returns a dictionary with:
        - top_risk_factors    : features INCREASING dropout risk
        - top_protective      : features DECREASING dropout risk
        - shap_values         : raw SHAP values for all features
        - feature_names       : all feature names

        Used by:
        - Student Dashboard (show student their own risk factors)
        - Teacher Dashboard (drill down into individual student)
        - Counseling Recommender (suggest actions based on top factors)
        """
        logger.info("Explaining individual student prediction...")
        try:
            if self.explainer is None:
                raise CustomException(
                    "Explainer not built. Call build_explainer() first.", sys
                )

            # Drop target if present
            if "Target" in student_df.columns:
                student_df = student_df.drop(columns=["Target"])

            # Transform student data
            X_transformed = self.preprocessor.transform(student_df)
            X_df = pd.DataFrame(X_transformed, columns=self.feature_names)

            # Get SHAP values
            # shap_values shape: (n_students, n_features, n_classes)
            # We take [:, :, 1] — SHAP values for Dropout class
            shap_vals = self.explainer.shap_values(X_df)

            # Shape is (n_samples, n_features, n_classes) — take dropout class [:, :, 1]
            if isinstance(shap_vals, list):
                shap_for_dropout = np.array(shap_vals[1])[0]
            elif np.array(shap_vals).ndim == 3:
                shap_for_dropout = np.array(shap_vals)[0, :, 1]
            else:
                shap_for_dropout = np.array(shap_vals)[0]

            # Build feature-SHAP pairs
            feature_shap = dict(zip(self.feature_names, shap_for_dropout))

            # Sort by absolute SHAP value
            sorted_features = sorted(
                feature_shap.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Split into risk factors (positive) and protective (negative)
            top_risk_factors = [
                {"feature": f, "shap_value": round(v, 4), "impact": "increases dropout risk"}
                for f, v in sorted_features if v > 0
            ][:5]   # top 5 risk factors

            top_protective = [
                {"feature": f, "shap_value": round(v, 4), "impact": "decreases dropout risk"}
                for f, v in sorted_features if v < 0
            ][:5]   # top 5 protective factors

            return {
                "top_risk_factors" : top_risk_factors,
                "top_protective"   : top_protective,
                "all_shap_values"  : feature_shap,
                "feature_names"    : self.feature_names,
            }

        except Exception as e:
            raise CustomException(e, sys)

    def plot_student_explanation(self, student_df: pd.DataFrame,
                                  student_id: str = "student") -> str:
        """
        Creates a horizontal bar chart showing top SHAP factors
        for a single student.

        Red bars  → factors pushing toward Dropout (bad)
        Blue bars → factors protecting from Dropout (good)

        Saves the plot and returns the file path.
        Used by both dashboards to display visual explanation.
        """
        logger.info(f"Plotting explanation for {student_id}...")
        try:
            explanation = self.explain_student(student_df)
            all_shap    = explanation["all_shap_values"]

            # Get top 10 features by absolute SHAP value
            top_10 = sorted(
                all_shap.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]

            features = [f for f, _ in top_10]
            values   = [v for _, v in top_10]
            colors   = ["#e74c3c" if v > 0 else "#3498db" for v in values]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])

            ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Value (impact on dropout probability)")
            ax.set_title(
                f"Why is {student_id} at risk?\n"
                f"Red = increases dropout risk | Blue = decreases dropout risk",
                fontsize=12, fontweight="bold"
            )

            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(
                self.config.plots_dir, f"{student_id}_explanation.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Plot saved: {plot_path}")
            return plot_path

        except Exception as e:
            raise CustomException(e, sys)

    def plot_global_importance(self, X_train_df: pd.DataFrame) -> str:
        """
        Creates a global feature importance plot showing which
        features matter MOST across ALL students.

        This is the big picture view:
        "What are the top factors that predict dropout in general?"

        Used in Teacher Dashboard and project README.
        """
        logger.info("Plotting global feature importance...")
        try:
            if "Target" in X_train_df.columns:
                X_train_df = X_train_df.drop(columns=["Target"])

            # Use a sample of 200 for speed (SHAP is compute-heavy)
            sample_df = X_train_df.sample(
                n=min(200, len(X_train_df)), random_state=42
            )
            X_transformed = self.preprocessor.transform(sample_df)
            X_df = pd.DataFrame(X_transformed, columns=self.feature_names)

            # Get SHAP values for all samples
            shap_vals = self.explainer.shap_values(X_df)
            # Shape is (n_samples, n_features, n_classes) — take dropout class
            if isinstance(shap_vals, list):
                shap_for_dropout = np.array(shap_vals[1])
            elif np.array(shap_vals).ndim == 3:
                shap_for_dropout = np.array(shap_vals)[:, :, 1]
            else:
                shap_for_dropout = np.array(shap_vals)

            # Mean absolute SHAP per feature
            mean_shap = np.abs(shap_for_dropout).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature"    : self.feature_names,
                "importance" : mean_shap
            }).sort_values("importance", ascending=False).head(15)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(
                importance_df["feature"][::-1],
                importance_df["importance"][::-1],
                color="#e74c3c"
            )
            ax.set_xlabel("Mean |SHAP Value| (average impact on dropout probability)")
            ax.set_title(
                "Top 15 Features Driving Student Dropout\n(Global SHAP Importance)",
                fontsize=12, fontweight="bold"
            )
            plt.tight_layout()

            plot_path = os.path.join(self.config.plots_dir, "global_importance.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Global importance plot saved: {plot_path}")
            return plot_path

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------------------------------------------------
# RUN DIRECTLY TO TEST
# python xai/explainer.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    # Load train data (needed to build explainer)
    train_df    = pd.read_csv(os.path.join(PROJECT_ROOT, "artifacts", "train.csv"))
    enrolled_df = pd.read_csv(os.path.join(PROJECT_ROOT, "artifacts", "enrolled_students.csv"))

    # Build explainer
    explainer    = DropoutExplainer()
    X_train_df   = explainer.build_explainer(train_df)

    # --- Global importance plot ---
    global_plot = explainer.plot_global_importance(train_df)
    print(f"\n✅ Global importance plot saved: {global_plot}")

    # --- Explain a single high-risk student ---
    student     = enrolled_df.iloc[[0]]   # first enrolled student
    explanation = explainer.explain_student(student)

    print("\n" + "=" * 55)
    print("  WHY IS THIS STUDENT AT RISK?")
    print("=" * 55)

    print("\n🔴 Top Risk Factors (pushing toward Dropout):")
    for item in explanation["top_risk_factors"]:
        print(f"   {item['feature']:<45} SHAP: {item['shap_value']:+.4f}")

    print("\n🔵 Top Protective Factors (pushing toward Graduate):")
    for item in explanation["top_protective"]:
        print(f"   {item['feature']:<45} SHAP: {item['shap_value']:+.4f}")

    # --- Individual student plot ---
    plot_path = explainer.plot_student_explanation(student, student_id="student_0")
    print(f"\n✅ Student explanation plot saved: {plot_path}")