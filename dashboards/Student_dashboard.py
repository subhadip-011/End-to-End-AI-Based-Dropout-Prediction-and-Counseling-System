import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.components.risk_classifier import RiskClassifier
from src.logger import logger

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Student Risk Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------------
# CUSTOM CSS — clean, professional look
# ---------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .high-risk   { background: #fdecea; border: 2px solid #e74c3c; }
    .medium-risk { background: #fef9e7; border: 2px solid #f39c12; }
    .low-risk    { background: #eafaf1; border: 2px solid #27ae60; }
    .metric-box  {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# LOAD RESOURCES — cached so they load only once
# ---------------------------------------------------------------
@st.cache_resource
def load_classifier():
    return RiskClassifier()

@st.cache_data
def load_enrolled_students():
    path = os.path.join(PROJECT_ROOT, "artifacts", "enrolled_students.csv")
    return pd.read_csv(path)

@st.cache_data
def load_risk_history():
    path = os.path.join(PROJECT_ROOT, "artifacts", "risk_history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def get_risk_color(risk_level: str) -> str:
    return {"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#27ae60"}.get(
        risk_level, "#95a5a6"
    )

def get_risk_css_class(risk_level: str) -> str:
    return {"High Risk": "high-risk", "Medium Risk": "medium-risk", "Low Risk": "low-risk"}.get(
        risk_level, ""
    )

def plot_risk_gauge(probability: float, risk_level: str) -> go.Figure:
    """
    Gauge chart showing dropout probability.
    Green zone = safe, Yellow = monitor, Red = danger.
    """
    color = get_risk_color(risk_level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        title={"text": "Dropout Risk Score", "font": {"size": 18}},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis"  : {"range": [0, 100], "tickwidth": 1},
            "bar"   : {"color": color},
            "steps" : [
                {"range": [0, 35],  "color": "#eafaf1"},
                {"range": [35, 65], "color": "#fef9e7"},
                {"range": [65, 100],"color": "#fdecea"},
            ],
            "threshold": {
                "line" : {"color": color, "width": 4},
                "value": round(probability * 100, 1),
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30))
    return fig

def plot_shap_bars(explanation: dict) -> go.Figure:
    """
    Horizontal bar chart showing top SHAP factors for the student.
    Red = increases risk, Blue = decreases risk.
    """
    risk_factors = explanation["top_risk_factors"]
    protective   = explanation["top_protective"]

    all_factors = risk_factors + protective
    all_factors = sorted(all_factors, key=lambda x: abs(x["shap_value"]), reverse=True)[:10]

    features = [f["feature"] for f in all_factors]
    values   = [f["shap_value"] for f in all_factors]
    colors   = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=features[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="Why is this student at risk? (SHAP Explanation)",
        xaxis_title="SHAP Value → positive = increases dropout risk",
        height=420,
        margin=dict(l=20, r=80, t=50, b=20),
        showlegend=False,
    )
    return fig

def plot_risk_trend(student_id: str, history: dict) -> go.Figure:
    """
    Line chart showing how the student's risk has changed over weeks.
    """
    entries = history.get(student_id, [])
    if not entries:
        return None

    weeks  = [e["week"] for e in entries]
    probs  = [e["probability"] * 100 for e in entries]
    levels = [e["risk_level"] for e in entries]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=probs,
        mode="lines+markers",
        name="Risk Score",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=8),
        hovertemplate="Week %{x}<br>Risk: %{y:.1f}%<extra></extra>",
    ))

    # Add threshold lines
    fig.add_hline(y=35, line_dash="dot", line_color="#27ae60",
                  annotation_text="Low/Medium boundary (35%)")
    fig.add_hline(y=65, line_dash="dot", line_color="#e74c3c",
                  annotation_text="Medium/High boundary (65%)")

    fig.update_layout(
        title="Your Risk Trend Over Time",
        xaxis_title="Week",
        yaxis_title="Dropout Risk (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(t=50, b=30),
    )
    return fig


# ---------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------
def main():
    # Header
    st.markdown(
        '<div class="main-header">🎓 Student Risk Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#7f8c8d;'>"
        "Understand your academic risk and get personalized guidance</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # Load data
    classifier   = load_classifier()
    enrolled_df  = load_enrolled_students()
    risk_history = load_risk_history()

    # ── Sidebar: Student Selection ──────────────────────────────
    st.sidebar.title("🎓 Student Portal")
    st.sidebar.markdown("---")

    total_students = len(enrolled_df)
    student_index  = st.sidebar.selectbox(
        "Select your Student ID",
        options=range(total_students),
        format_func=lambda x: f"Student {x + 1:04d}",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 This dashboard shows your personal dropout risk score "
        "and explains the key factors affecting your academic journey."
    )

    # Get selected student data
    student_row = enrolled_df.iloc[[student_index]]
    student_id  = f"student_{student_index}"

    # Classify student
    result      = classifier.classify(student_row)
    probability = result["dropout_probability"].iloc[0]
    risk_level  = result["risk_level"].iloc[0]
    risk_emoji  = result["risk_emoji"].iloc[0]
    urgency     = result["counseling_urgency"].iloc[0]

    # ── Row 1: Risk Overview ────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.plotly_chart(
            plot_risk_gauge(probability, risk_level),
            use_container_width=True
        )

    with col2:
        css_class = get_risk_css_class(risk_level)
        st.markdown(f"""
        <div class="risk-card {css_class}">
            <h1>{risk_emoji}</h1>
            <h2>{risk_level}</h2>
            <p>Your current risk level</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        color = get_risk_color(risk_level)
        st.markdown(f"""
        <div class="metric-box" style="border-left: 4px solid {color}; margin-top: 1rem;">
            <h4>📋 Recommended Action</h4>
            <p style="color:{color}; font-weight:600;">{urgency}</p>
        </div>
        <div class="metric-box" style="margin-top: 1rem;">
            <h4>📊 Risk Score</h4>
            <p style="font-size: 2rem; font-weight:bold; color:{color};">
                {round(probability * 100, 1)}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Row 2: SHAP Explanation ─────────────────────────────────
    st.markdown('<div class="section-header">🔍 Why are you at this risk level?</div>',
                unsafe_allow_html=True)

    try:
        # Build SHAP explainer
        from xai.explainer import DropoutExplainer
        train_df  = pd.read_csv(os.path.join(PROJECT_ROOT, "artifacts", "train.csv"))
        explainer = DropoutExplainer()
        explainer.build_explainer(train_df)
        explanation = explainer.explain_student(student_row)

        col_shap, col_tips = st.columns([1.5, 1])

        with col_shap:
            st.plotly_chart(
                plot_shap_bars(explanation),
                use_container_width=True
            )

        with col_tips:
            st.markdown("**🔴 Top Risk Factors**")
            for item in explanation["top_risk_factors"][:3]:
                st.error(f"📌 **{item['feature']}**  \nImpact: `{item['shap_value']:+.3f}`")

            st.markdown("**🔵 Protective Factors**")
            for item in explanation["top_protective"][:3]:
                st.success(f"✅ **{item['feature']}**  \nImpact: `{item['shap_value']:+.3f}`")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    st.divider()

    # ── Row 3: Risk Trend ───────────────────────────────────────
    st.markdown('<div class="section-header">📈 Your Risk Trend Over Time</div>',
                unsafe_allow_html=True)

    trend_fig = plot_risk_trend(student_id, risk_history)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info(
            "📭 No trend history yet. "
            "Run the Early Warning System to generate weekly snapshots."
        )

    st.divider()

    # ── Row 4: Student Profile ───────────────────────────────────
    st.markdown('<div class="section-header">👤 Your Academic Profile</div>',
                unsafe_allow_html=True)

    profile_cols = [
        "Age at enrollment", "Gender", "Scholarship holder",
        "Debtor", "Tuition fees up to date",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    ]

    # Only show columns that exist in the data
    available = [c for c in profile_cols if c in student_row.columns]
    profile_df = student_row[available].T.reset_index()
    profile_df.columns = ["Feature", "Your Value"]
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # ── Footer ──────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#bdc3c7; font-size:0.85rem;'>"
        "🔒 Your data is confidential. "
        "Powered by XGBoost + SHAP | Dropout Prediction System</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()