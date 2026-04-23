import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


# ---------------------------------------------------------------
# CONFIG: all file paths in one place
# If you want to change any path, change it here only
# ---------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


# ---------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads the raw CSV file.
        Our dataset uses semicolon (;) as separator — not comma.
        """
        logger.info("Loading raw dataset...")
        df = pd.read_csv(file_path, sep=";")
        logger.info(f"Dataset loaded. Shape: {df.shape}")
        return df

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strips extra spaces and tab characters from column names.
        Our dataset has a tab character in 'Daytime/evening attendance'.
        """
        df.columns = df.columns.str.strip()
        logger.info("Column names cleaned.")
        return df

    def _filter_enrolled(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        KEY STEP — Remove 'Enrolled' students from training data.

        Why? Because Enrolled students have unknown final outcome.
        We don't know if they will Graduate or Dropout.
        Training on unknown labels would introduce noise into the model.

        We keep them separately for PREDICTION only (Early Warning System).
        """
        enrolled_df = df[df["Target"] == "Enrolled"].copy()
        training_df = df[df["Target"] != "Enrolled"].copy()

        logger.info(f"Total students       : {len(df)}")
        logger.info(f"Training set (known) : {len(training_df)}  (Graduate + Dropout)")
        logger.info(f"Enrolled students    : {len(enrolled_df)}  (saved for prediction)")

        # Save enrolled students separately — Early Warning System will use them
        enrolled_path = os.path.join("artifacts", "enrolled_students.csv")
        enrolled_df.to_csv(enrolled_path, index=False)
        logger.info(f"Enrolled students saved to: {enrolled_path}")

        return training_df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert text labels to binary numbers.
        Dropout  → 1  (positive class — what we want to predict)
        Graduate → 0  (negative class)
        """
        df["Target"] = df["Target"].map({"Dropout": 1, "Graduate": 0})
        logger.info("Target encoded: Dropout=1, Graduate=0")
        logger.info(f"Target distribution:\n{df['Target'].value_counts()}")
        return df

    def initiate_data_ingestion(self, file_path: str):
        """
        Main method — runs all steps in order.
        Returns paths to train and test CSV files.
        """
        logger.info(" Data Ingestion Started ")
        try:
            # Step 1: Load
            df = self._load_data(file_path)

            # Step 2: Clean column names
            df = self._clean_column_names(df)

            # Step 3: Remove Enrolled students from training
            df = self._filter_enrolled(df)

            # Step 4: Encode target column
            df = self._encode_target(df)

            # Step 5: Save raw (filtered) data
            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw filtered data saved to: {self.config.raw_data_path}")

            # Step 6: Train-Test Split (80/20, stratified so class ratio is preserved)
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["Target"]   # ensures both splits have same Dropout ratio
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Train set : {train_df.shape} → {self.config.train_data_path}")
            logger.info(f"Test set  : {test_df.shape}  → {self.config.test_data_path}")
            logger.info("Data Ingestion Completed")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------------------------------------------------
# RUN DIRECTLY TO TEST
# python src/components/data_ingestion.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion("notebook/data/student_data.csv")
    print(f"\nTrain: {train_path}")
    print(f"Test : {test_path}")