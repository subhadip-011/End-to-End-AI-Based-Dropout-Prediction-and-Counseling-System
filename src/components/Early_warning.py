import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.components.risk_classifier import RiskClassifier


# ---------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# ---------------------------------------------------------------
# HOW THE EARLY WARNING SYSTEM WORKS
#
# In real world, student data is updated every semester/week.
# Each time new data comes in, we:
#   1. Predict risk score for each enrolled student
#   2. Compare with their PREVIOUS risk score
#   3. If risk is RISING → trigger an alert
#   4. Save history so we can track trends over time
#
# Since our dataset is static (no live updates),
# we SIMULATE multiple time snapshots by adding small realistic
# noise to features — mimicking how student performance
# fluctuates week to week.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
@dataclass
class EarlyWarningConfig:
    history_path: str  = os.path.join(PROJECT_ROOT, "artifacts", "risk_history.json")
    alerts_path: str   = os.path.join(PROJECT_ROOT, "artifacts", "alerts.csv")
    snapshot_path: str = os.path.join(PROJECT_ROOT, "artifacts", "latest_snapshot.csv")


# ---------------------------------------------------------------
# ALERT THRESHOLDS
# ---------------------------------------------------------------
RISK_RISE_THRESHOLD  = 0.10   # if probability rises by 10% → alert
CRITICAL_PROBABILITY = 0.80   # if probability crosses 80% → critical alert


# ---------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------
class EarlyWarningSystem:
    def __init__(self):
        self.config     = EarlyWarningConfig()
        self.classifier = RiskClassifier()
        self.history    = self._load_history()

    def _load_history(self) -> dict:
        """
        Load existing risk history from JSON file.
        If no history exists yet, start fresh.

        History structure:
        {
            "student_0": [
                {"date": "2024-01-01", "probability": 0.45, "risk_level": "Medium Risk"},
                {"date": "2024-01-08", "probability": 0.67, "risk_level": "High Risk"},
            ],
            ...
        }
        """
        if os.path.exists(self.config.history_path):
            with open(self.config.history_path, "r") as f:
                history = json.load(f)
            logger.info(f"Loaded history for {len(history)} students")
        else:
            history = {}
            logger.info("No history found. Starting fresh.")
        return history

    def _save_history(self):
        """Save updated risk history back to JSON."""
        with open(self.config.history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"History saved: {self.config.history_path}")

    def _simulate_new_snapshot(self, df: pd.DataFrame, week: int) -> pd.DataFrame:
        """
        Simulates weekly data update by adding small noise to
        continuous features like grades and attendance.

        WHY simulate?
        Our dataset is static — it has no weekly updates.
        In production, this method would be replaced by fetching
        fresh data from the university's student information system.

        The noise mimics realistic week-to-week fluctuation in:
        - Grades (go up or down slightly)
        - Units approved (accumulate over semester)
        """
        snapshot = df.copy()

        # Features that realistically change week to week
        fluctuating_features = [
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 2nd sem (evaluations)",
        ]

        np.random.seed(week)   # different seed each week = different noise

        for col in fluctuating_features:
            if col in snapshot.columns:
                noise = np.random.normal(
                    loc=0,      # centered around 0 (no bias)
                    scale=0.3,  # small fluctuation
                    size=len(snapshot)
                )
                snapshot[col] = np.clip(snapshot[col] + noise, 0, 20)

        return snapshot

    def _check_alerts(
        self,
        student_id: str,
        current_prob: float,
        current_risk: str
    ) -> dict | None:
        """
        Checks if an alert should be triggered for a student.

        Two types of alerts:
        1. RISING RISK   — probability increased by 10%+ since last check
        2. CRITICAL RISK — probability crossed 80% threshold
        """
        if student_id not in self.history:
            return None   # first snapshot, no previous to compare

        previous_entries = self.history[student_id]
        if not previous_entries:
            return None

        previous_prob  = previous_entries[-1]["probability"]
        previous_risk  = previous_entries[-1]["risk_level"]
        prob_change    = current_prob - previous_prob

        alert = None

        # Alert Type 1: Risk is rising significantly
        if prob_change >= RISK_RISE_THRESHOLD:
            alert = {
                "student_id"    : student_id,
                "alert_type"    : "RISING RISK",
                "previous_prob" : round(previous_prob, 4),
                "current_prob"  : round(current_prob, 4),
                "change"        : round(prob_change, 4),
                "previous_risk" : previous_risk,
                "current_risk"  : current_risk,
                "message"       : (
                    f"Risk increased by {round(prob_change*100, 1)}% "
                    f"({previous_risk} → {current_risk})"
                ),
                "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        # Alert Type 2: Crossed critical threshold
        elif current_prob >= CRITICAL_PROBABILITY and previous_prob < CRITICAL_PROBABILITY:
            alert = {
                "student_id"    : student_id,
                "alert_type"    : "CRITICAL RISK",
                "previous_prob" : round(previous_prob, 4),
                "current_prob"  : round(current_prob, 4),
                "change"        : round(prob_change, 4),
                "previous_risk" : previous_risk,
                "current_risk"  : current_risk,
                "message"       : (
                    f"Student crossed CRITICAL threshold! "
                    f"Dropout probability: {round(current_prob*100, 1)}%"
                ),
                "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        return alert

    def run_snapshot(self, enrolled_df: pd.DataFrame, week: int = 1):
        """
        Main method — runs a full weekly snapshot cycle:
        1. Simulate new week's data
        2. Classify all students
        3. Check for alerts
        4. Update history
        5. Save alerts

        Parameters:
        - enrolled_df : DataFrame of currently enrolled students
        - week        : week number (1, 2, 3...) for simulation
        """
        logger.info(f"========== Early Warning Snapshot — Week {week} ==========")
        try:
            # Step 1: Simulate this week's data
            snapshot_df = self._simulate_new_snapshot(enrolled_df, week)
            logger.info(f"Snapshot created for week {week} — {len(snapshot_df)} students")

            # Step 2: Classify all students
            classified_df = self.classifier.classify(snapshot_df)

            # Step 3: Check alerts and update history
            alerts     = []
            today      = datetime.now().strftime("%Y-%m-%d")

            for idx, row in classified_df.iterrows():
                student_id   = f"student_{idx}"
                current_prob = row["dropout_probability"]
                current_risk = row["risk_level"]

                # Check if alert needed
                alert = self._check_alerts(student_id, current_prob, current_risk)
                if alert:
                    alerts.append(alert)

                # Update history
                if student_id not in self.history:
                    self.history[student_id] = []

                self.history[student_id].append({
                    "week"        : week,
                    "date"        : today,
                    "probability" : round(current_prob, 4),
                    "risk_level"  : current_risk,
                })

            # Step 4: Save history
            self._save_history()

            # Step 5: Save snapshot
            classified_df.to_csv(self.config.snapshot_path, index=False)

            # Step 6: Save alerts
            if alerts:
                alerts_df = pd.DataFrame(alerts)
                alerts_df.to_csv(self.config.alerts_path, index=False)
                logger.warning(f"⚠️  {len(alerts)} alerts triggered!")
            else:
                logger.info("No alerts triggered this week.")

            logger.info(f"========== Snapshot Week {week} Complete ==========")

            return classified_df, alerts

        except Exception as e:
            raise CustomException(e, sys)

    def get_student_trend(self, student_id: str) -> list:
        """
        Returns the full risk history for a single student.
        Used by Student Dashboard to show trend graph.
        """
        return self.history.get(student_id, [])

    def get_all_alerts(self) -> pd.DataFrame:
        """Returns all saved alerts. Used by Teacher Dashboard."""
        if os.path.exists(self.config.alerts_path):
            return pd.read_csv(self.config.alerts_path)
        return pd.DataFrame()


# ---------------------------------------------------------------
# RUN DIRECTLY TO TEST
# python src/components/early_warning.py
# ---------------------------------------------------------------
if __name__ == "__main__":

    # Load enrolled students
    enrolled_path = os.path.join(PROJECT_ROOT, "artifacts", "enrolled_students.csv")
    enrolled_df   = pd.read_csv(enrolled_path)
    logger.info(f"Loaded {len(enrolled_df)} enrolled students")

    # Create Early Warning System
    ews = EarlyWarningSystem()

    # Simulate 3 weekly snapshots
    for week in range(1, 4):
        print(f"\n{'='*50}")
        print(f"  RUNNING WEEK {week} SNAPSHOT")
        print(f"{'='*50}")

        classified_df, alerts = ews.run_snapshot(enrolled_df, week=week)

        # Risk summary for this week
        risk_counts = classified_df["risk_level"].value_counts()
        print(f"\nRisk Distribution — Week {week}:")
        for level, count in risk_counts.items():
            print(f"  {level}: {count} students")

        # Show alerts
        if alerts:
            print(f"\n⚠️  {len(alerts)} Alerts Triggered:")
            for alert in alerts[:3]:   # show first 3
                print(f"  [{alert['alert_type']}] {alert['message']}")
        else:
            print("\n✅ No alerts this week.")

    # Show trend for one student
    print(f"\n{'='*50}")
    print("  SAMPLE STUDENT TREND (student_0)")
    print(f"{'='*50}")
    trend = ews.get_student_trend("student_0")
    for entry in trend:
        print(
            f"  Week {entry['week']} | "
            f"Probability: {entry['probability']} | "
            f"Risk: {entry['risk_level']}"
        )