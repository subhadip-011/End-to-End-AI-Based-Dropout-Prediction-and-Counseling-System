import os
import sys
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
# HOW THE COUNSELING RECOMMENDER WORKS
#
# Input  : SHAP explanation (top risk factors for a student)
# Process: Match each risk factor to a rule-based recommendation
# Output : Personalized action plan for that specific student
#
# WHY rule-based and not another ML model?
# - Recommendations must be EXPLAINABLE to teachers and students
# - "You have low 2nd sem grades → attend tutoring" is clear
# - A second ML model would be a black box on top of a black box
# - Rule-based systems are used in real EdTech products (Coursera,
#   Khan Academy) for exactly this reason — trust and transparency
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# RECOMMENDATION RULES
# Each rule maps a SHAP feature keyword → recommendation dict
#
# Structure:
# "keyword in feature name" → {
#     "issue"      : what the problem is
#     "action"     : what the student should do
#     "resource"   : where to get help
#     "priority"   : how urgent (High / Medium / Low)
#     "timeline"   : when to act
# }
# ---------------------------------------------------------------
RECOMMENDATION_RULES = [
    {
        "keywords"  : ["2nd sem (approved)", "2nd sem (grade)"],
        "issue"     : "Poor performance in 2nd semester",
        "action"    : "Attend extra tutoring sessions and form study groups. "
                      "Focus on understanding concepts rather than memorization.",
        "resource"  : "Academic Tutoring Center | Office Hours with Professor",
        "priority"  : "High",
        "timeline"  : "Immediate — this week",
        "icon"      : "📚",
    },
    {
        "keywords"  : ["1st sem (approved)", "1st sem (grade)"],
        "issue"     : "Weak foundation from 1st semester",
        "action"    : "Review 1st semester material. "
                      "Identify specific topics where you are struggling "
                      "and seek targeted help.",
        "resource"  : "Peer Tutoring Program | Online Study Resources",
        "priority"  : "High",
        "timeline"  : "Within 1 week",
        "icon"      : "📖",
    },
    {
        "keywords"  : ["tuition fees", "debtor"],
        "issue"     : "Financial difficulties affecting studies",
        "action"    : "Apply for emergency financial aid or student loans. "
                      "Financial stress is a major dropout driver — "
                      "do not delay seeking help.",
        "resource"  : "Financial Aid Office | Student Welfare Fund | Scholarship Portal",
        "priority"  : "High",
        "timeline"  : "Immediate — contact financial aid today",
        "icon"      : "💰",
    },
    {
        "keywords"  : ["scholarship"],
        "issue"     : "Scholarship status at risk",
        "action"    : "Review scholarship requirements and academic conditions. "
                      "Speak with your scholarship coordinator to understand "
                      "what grades you need to maintain.",
        "resource"  : "Scholarship Office | Academic Advisor",
        "priority"  : "High",
        "timeline"  : "Within 3 days",
        "icon"      : "🏆",
    },
    {
        "keywords"  : ["age at enrollment"],
        "issue"     : "Age-related academic transition challenges",
        "action"    : "Connect with a counselor to discuss time management "
                      "and balancing responsibilities. Older students often "
                      "have work or family commitments that need addressing.",
        "resource"  : "Student Counseling Center | Time Management Workshop",
        "priority"  : "Medium",
        "timeline"  : "Within 2 weeks",
        "icon"      : "🕐",
    },
    {
        "keywords"  : ["displaced"],
        "issue"     : "Relocation or displacement affecting stability",
        "action"    : "Reach out to student support services for housing "
                      "and community resources. Feeling settled is important "
                      "for academic focus.",
        "resource"  : "Student Support Services | Housing Office",
        "priority"  : "Medium",
        "timeline"  : "Within 1 week",
        "icon"      : "🏠",
    },
    {
        "keywords"  : ["attendance", "daytime", "evening"],
        "issue"     : "Attendance or schedule mismatch",
        "action"    : "Review your course schedule and ensure it fits your "
                      "lifestyle. Consider switching between day/evening "
                      "classes if conflicts exist.",
        "resource"  : "Academic Registrar | Course Advisor",
        "priority"  : "Medium",
        "timeline"  : "Before next semester registration",
        "icon"      : "🗓️",
    },
    {
        "keywords"  : ["curricular units", "enrolled", "evaluations"],
        "issue"     : "Course load or engagement issues",
        "action"    : "Evaluate if you are taking too many or too few units. "
                      "Overloading leads to burnout; underloading may signal "
                      "disengagement.",
        "resource"  : "Academic Advisor | Course Load Calculator",
        "priority"  : "Medium",
        "timeline"  : "Within 2 weeks",
        "icon"      : "⚖️",
    },
    {
        "keywords"  : ["previous qualification", "admission grade"],
        "issue"     : "Academic foundation below course requirements",
        "action"    : "Enroll in bridging or foundation courses to strengthen "
                      "your academic base. Early intervention prevents later failure.",
        "resource"  : "Academic Bridging Program | Foundation Course Office",
        "priority"  : "Medium",
        "timeline"  : "Start of next term",
        "icon"      : "🎯",
    },
    {
        "keywords"  : ["unemployment", "gdp", "inflation"],
        "issue"     : "External economic pressures",
        "action"    : "Explore part-time work-study programs on campus. "
                      "Economic uncertainty can be managed with proper "
                      "financial planning and institutional support.",
        "resource"  : "Career Services | Work-Study Program | Student Financial Planning",
        "priority"  : "Low",
        "timeline"  : "Within 1 month",
        "icon"      : "📊",
    },
    {
        "keywords"  : ["gender", "international", "nacionality"],
        "issue"     : "Social or cultural adjustment challenges",
        "action"    : "Connect with cultural student groups and international "
                      "student services. Building a support network improves "
                      "academic persistence.",
        "resource"  : "International Student Office | Cultural Student Associations",
        "priority"  : "Low",
        "timeline"  : "Within 1 month",
        "icon"      : "🌍",
    },
]

# ---------------------------------------------------------------
# GENERAL RECOMMENDATIONS (always shown regardless of risk factors)
# ---------------------------------------------------------------
GENERAL_RECOMMENDATIONS = {
    "High Risk": [
        {
            "action"   : "Schedule an urgent counseling appointment this week.",
            "resource" : "Student Counseling Center",
            "icon"     : "🆘",
        },
        {
            "action"   : "Inform your course coordinator about your situation.",
            "resource" : "Department Head / Course Coordinator",
            "icon"     : "📞",
        },
    ],
    "Medium Risk": [
        {
            "action"   : "Schedule a check-in with your academic advisor in the next 2 weeks.",
            "resource" : "Academic Advising Office",
            "icon"     : "📅",
        },
    ],
    "Low Risk": [
        {
            "action"   : "Keep up the good work! Review your progress monthly.",
            "resource" : "Self-monitoring via Student Dashboard",
            "icon"     : "✅",
        },
    ],
}


# ---------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------
class CounselingRecommender:

    def get_recommendations(
        self,
        explanation: dict,
        risk_level: str,
        risk_score: float,
    ) -> dict:
        """
        Generates a personalized counseling plan based on
        SHAP explanation and risk level.

        Parameters:
        - explanation : output from DropoutExplainer.explain_student()
        - risk_level  : "High Risk" / "Medium Risk" / "Low Risk"
        - risk_score  : dropout probability (0 to 1)

        Returns a dict with:
        - personalized_recommendations : matched to student's top SHAP factors
        - general_recommendations      : based on risk level
        - summary                      : one-line counselor summary
        - priority_action              : the single most important thing to do
        """
        logger.info(f"Generating recommendations for {risk_level} student...")

        try:
            personalized = []
            matched_keywords = set()

            # Get top risk factors from SHAP explanation
            top_factors = explanation.get("top_risk_factors", [])

            for factor in top_factors:
                feature_name = factor["feature"].lower()
                shap_value   = factor["shap_value"]

                # Match feature to recommendation rules
                for rule in RECOMMENDATION_RULES:
                    # Check if any keyword matches this feature
                    if any(kw.lower() in feature_name for kw in rule["keywords"]):
                        # Avoid duplicate recommendations
                        rule_key = rule["issue"]
                        if rule_key not in matched_keywords:
                            matched_keywords.add(rule_key)
                            personalized.append({
                                "issue"      : rule["issue"],
                                "action"     : rule["action"],
                                "resource"   : rule["resource"],
                                "priority"   : rule["priority"],
                                "timeline"   : rule["timeline"],
                                "icon"       : rule["icon"],
                                "shap_impact": round(shap_value, 4),
                                "feature"    : factor["feature"],
                            })
                        break   # one rule per factor

            # Sort by priority
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            personalized.sort(key=lambda x: priority_order.get(x["priority"], 3))

            # Get general recommendations for risk level
            general = GENERAL_RECOMMENDATIONS.get(risk_level, [])

            # Build summary
            summary = self._build_summary(
                risk_level, risk_score, personalized
            )

            # Priority action = first high priority item
            priority_action = None
            for rec in personalized:
                if rec["priority"] == "High":
                    priority_action = rec
                    break
            if not priority_action and general:
                priority_action = general[0]

            logger.info(
                f"Generated {len(personalized)} personalized + "
                f"{len(general)} general recommendations."
            )

            return {
                "personalized_recommendations": personalized,
                "general_recommendations"     : general,
                "summary"                     : summary,
                "priority_action"             : priority_action,
                "risk_level"                  : risk_level,
                "risk_score"                  : risk_score,
            }

        except Exception as e:
            raise CustomException(e, sys)

    def _build_summary(
        self,
        risk_level: str,
        risk_score: float,
        recommendations: list
    ) -> str:
        """Builds a one-paragraph counselor summary."""

        score_pct = round(risk_score * 100, 1)
        n_issues  = len(recommendations)
        top_issue = recommendations[0]["issue"] if recommendations else "general performance"

        if risk_level == "High Risk":
            return (
                f"This student has a dropout probability of {score_pct}% — "
                f"classified as High Risk. Immediate intervention is required. "
                f"The primary concern is {top_issue}. "
                f"{n_issues} specific issue(s) have been identified. "
                f"An urgent counseling session should be scheduled this week."
            )
        elif risk_level == "Medium Risk":
            return (
                f"This student has a dropout probability of {score_pct}% — "
                f"classified as Medium Risk. Proactive monitoring is advised. "
                f"The main concern is {top_issue}. "
                f"A check-in within the next 2 weeks is recommended."
            )
        else:
            return (
                f"This student has a dropout probability of {score_pct}% — "
                f"classified as Low Risk. "
                f"Continue monitoring monthly and encourage current study habits."
            )

    def format_for_display(self, recommendations: dict) -> str:
        """
        Formats the full recommendation plan as plain text.
        Used for email body or PDF report generation.
        """
        lines = []
        lines.append("=" * 55)
        lines.append("   PERSONALIZED COUNSELING PLAN")
        lines.append("=" * 55)
        lines.append(f"\nSUMMARY:\n{recommendations['summary']}\n")

        if recommendations["priority_action"]:
            pa = recommendations["priority_action"]
            lines.append(f"⚡ PRIORITY ACTION:\n   {pa.get('action','')}\n")

        lines.append("PERSONALIZED RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations["personalized_recommendations"], 1):
            lines.append(f"\n{i}. {rec['icon']} {rec['issue']}")
            lines.append(f"   Priority : {rec['priority']}")
            lines.append(f"   Action   : {rec['action']}")
            lines.append(f"   Resource : {rec['resource']}")
            lines.append(f"   Timeline : {rec['timeline']}")

        lines.append("\nGENERAL RECOMMENDATIONS:")
        for rec in recommendations["general_recommendations"]:
            lines.append(f"  {rec['icon']} {rec['action']}")
            lines.append(f"     → {rec['resource']}")

        return "\n".join(lines)


# ---------------------------------------------------------------
# RUN DIRECTLY TO TEST
# python counseling/recommender.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import dill

    # Load model artifacts
    with open(os.path.join(PROJECT_ROOT, "artifacts", "model.pkl"), "rb") as f:
        model = dill.load(f)
    with open(os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl"), "rb") as f:
        preprocessor = dill.load(f)

    # Load a high risk enrolled student
    enrolled_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "artifacts", "enrolled_students.csv")
    )
    train_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "artifacts", "train.csv")
    )

    # Get SHAP explanation
    import importlib.util
    for name in ["Explainer_shap.py", "explainer.py"]:
        path = os.path.join(PROJECT_ROOT, "xai", name)
        if os.path.exists(path):
            spec   = importlib.util.spec_from_file_location("explainer", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            DropoutExplainer = module.DropoutExplainer
            break

    # Classify and explain a student
    from src.components.risk_classifier import RiskClassifier
    classifier  = RiskClassifier()
    student     = enrolled_df.iloc[[0]]
    result      = classifier.classify(student)
    risk_level  = result["risk_level"].iloc[0]
    risk_score  = result["dropout_probability"].iloc[0]

    explainer   = DropoutExplainer()
    explainer.build_explainer(train_df)
    explanation = explainer.explain_student(student)

    # Generate recommendations
    recommender     = CounselingRecommender()
    recommendations = recommender.get_recommendations(
        explanation, risk_level, risk_score
    )

    # Print formatted plan
    print(recommender.format_for_display(recommendations))