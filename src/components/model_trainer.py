import os
import sys
import numpy as np
import pandas as pd
import dill

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logger
from src.exception import CustomException


# ---------------------------------------------------------------
# CONFIG — all output paths in one place
# ---------------------------------------------------------------
@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ---------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def _get_feature_groups(self):
        """
        Split features into two groups:
        1. Continuous — need StandardScaler (grades, age, GDP etc.)
        2. Binary/Categorical — already 0/1, no scaling needed

        WHY separate them?
        If we scale binary columns like Gender (0/1),
        StandardScaler turns them into decimals like -0.7, 1.2
        which destroys their meaning.
        """

        continuous_features = [
            "Previous qualification (grade)",
            "Admission grade",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Age at enrollment",
            "Unemployment rate",
            "Inflation rate",
            "GDP",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (evaluations)",
        ]

        binary_or_categorical_features = [
            "Marital status",
            "Application mode",
            "Application order",
            "Course",
            "Daytime/evening attendance",
            "Previous qualification",
            "Nacionality",
            "Mother's qualification",
            "Father's qualification",
            "Mother's occupation",
            "Father's occupation",
            "Displaced",
            "Educational special needs",
            "Debtor",
            "Tuition fees up to date",
            "Gender",
            "Scholarship holder",
            "International",
            "Curricular units 1st sem (credited)",
            "Curricular units 1st sem (without evaluations)",
            "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (without evaluations)",
        ]

        return continuous_features, binary_or_categorical_features

    def _build_preprocessor(self, continuous_features, binary_features):
        """
        Builds a Scikit-learn ColumnTransformer pipeline.

        ColumnTransformer lets us apply DIFFERENT transformations
        to DIFFERENT columns — all in one reusable object.

        Pipeline is industry standard because:
        - fit() on train, transform() on test (no data leakage)
        - Save as .pkl and reuse in production
        - New data gets same transformation automatically
        """

        # Only scale continuous features
        continuous_pipeline = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        # Binary/categorical features pass through unchanged
        categorical_pipeline = Pipeline(steps=[
            ("passthrough", "passthrough")
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("continuous", continuous_pipeline, continuous_features),
            ("categorical", categorical_pipeline, binary_features),
        ])

        return preprocessor

    def _save_object(self, obj, path):
        """
        Saves any Python object (model, preprocessor) as a .pkl file.
        Uses dill instead of pickle — dill handles complex objects better.
        """
        with open(path, "wb") as f:
            dill.dump(obj, f)
        logger.info(f"Object saved to: {path}")

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Main method:
        1. Load train and test data
        2. Separate features (X) and target (y)
        3. Fit preprocessor on train only
        4. Transform both train and test
        5. Save preprocessor as .pkl
        6. Return transformed arrays
        """
        logger.info("========== Data Transformation Started ==========")
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            # Get feature groups
            continuous_features, binary_features = self._get_feature_groups()

            # Separate X and y
            target_col = "Target"

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            logger.info(f"Features: {X_train.shape[1]} | Target distribution (train): {y_train.value_counts().to_dict()}")

            # Build preprocessor
            preprocessor = self._build_preprocessor(continuous_features, binary_features)

            # IMPORTANT: fit ONLY on train — then transform both
            # Fitting on test would cause DATA LEAKAGE
            logger.info("Fitting preprocessor on train data only (preventing data leakage)...")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)

            logger.info(f"Transformed train shape: {X_train_transformed.shape}")
            logger.info(f"Transformed test shape : {X_test_transformed.shape}")

            # Combine X and y back into arrays for model trainer
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr  = np.c_[X_test_transformed,  np.array(y_test)]

            # Save preprocessor — needed later for prediction pipeline
            self._save_object(preprocessor, self.config.preprocessor_path)

            logger.info("========== Data Transformation Completed ==========")

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------------------------------------------------
# RUN DIRECTLY TO TEST
# python src/components/data_transformation.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # First run ingestion to get train/test paths
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion("data.csv")

    # Then run transformation
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        train_path, test_path
    )

    print(f"\nTrain array shape : {train_arr.shape}")
    print(f"Test array shape  : {test_arr.shape}")
    print(f"Preprocessor saved: {preprocessor_path}")