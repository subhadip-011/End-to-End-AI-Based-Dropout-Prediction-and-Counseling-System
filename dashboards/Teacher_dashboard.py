import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

# Load .env credentials before any imports that need them
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------
# LOCAL IMPORTS
# ---------------------------------------------------------------
from src.components.risk_classifier import RiskClassifier
from src.components.alert_system import AlertSystem
from src.logger import logger

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Teacher Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: bold;
        color: #2c3e50; text-align: center; padding: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1.2rem; text-align: center; border-top: 4px solid;
    }
    .card-red    { border-color: #e74c3c; }
    .card-yellow { border-color: #f39c12; }
    .card-green  { border-color: #27ae60; }
    .card-blue   { border-color: #3498db; }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #2c3e50;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 0.3rem; margin: 1.5rem 0 1rem 0;
    }
    .alert-box {
        background: #fdecea; border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0;
    }
    .alert-rising {
        background: #fef9e7; border-left: 4px solid #f39c12;
    }
    .send-alert-box {
        background: #eaf4fb; border: 1px solid #3498db;
        border-radius: 12px; padding: 1.5rem; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------
@st.cache_resource
def load_classifier():
    return RiskClassifier()

@st.cache_data
def load_and_classify():
    path = os.path.join(PROJECT_ROOT, "artifacts", "enrolled_students.csv")
    df   = pd.read_csv(path)
    clf  = RiskClassifier()
    return clf.classify(df)

@st.cache_data
def load_risk_history():
    path = os.path.join(PROJECT_ROOT, "artifacts", "risk_history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_alerts():
    path = os.path.join(PROJECT_ROOT, "artifacts", "alerts.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_train_data():
    return pd.read_csv(os.path.join(PROJECT_ROOT, "artifacts", "train.csv"))


# ---------------------------------------------------------------
# LOAD SHAP EXPLAINER SAFELY
# ---------------------------------------------------------------
def load_explainer():
    import importlib.util
    for name in ["Explainer_shap.py", "explainer.py"]:
        path = os.path.join(PROJECT_ROOT, "xai", name)
        if os.path.exists(path):
            spec   = importlib.util.spec_from_file_location("explainer", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.DropoutExplainer
    raise FileNotFoundError("No explainer file found in xai/ folder.")


# ---------------------------------------------------------------
# PLOT HELPERS
# ---------------------------------------------------------------
def get_risk_color(risk_level):
    return {
        "High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#27ae60"
    }.get(risk_level, "#95a5a6")

def plot_donut(result_df):
    counts = result_df["risk_level"].value_counts()
    colors = {
        "High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#27ae60"
    }
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        marker_colors=[colors.get(l, "#95a5a6") for l in counts.index],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} students<extra></extra>",
    ))
    fig.update_layout(title="Class Risk Distribution",
                      height=320, margin=dict(t=50, b=20, l=20, r=20))
    return fig

def plot_histogram(result_df):
    fig = px.histogram(
        result_df, x="dropout_probability", nbins=30,
        color="risk_level",
        color_discrete_map={
            "High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#27ae60"
        },
        title="Distribution of Dropout Probabilities",
    )
    fig.add_vline(x=0.35, line_dash="dash", line_color="#27ae60",
                  annotation_text="35%")
    fig.add_vline(x=0.65, line_dash="dash", line_color="#e74c3c",
                  annotation_text="65%")
    fig.update_layout(height=320, margin=dict(t=50, b=30))
    return fig

def plot_trend(history):
    if not history:
        return None
    week_data = {}
    for entries in history.values():
        for e in entries:
            w = e["week"]
            if w not in week_data:
                week_data[w] = []
            week_data[w].append(e["probability"])

    weeks    = sorted(week_data.keys())
    avg_prob = [np.mean(week_data[w]) * 100 for w in weeks]
    high_pct = [
        sum(1 for p in week_data[w] if p >= 0.65) / len(week_data[w]) * 100
        for w in weeks
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=avg_prob, name="Avg Risk %",
                             mode="lines+markers",
                             line=dict(color="#3498db", width=2)))
    fig.add_trace(go.Scatter(x=weeks, y=high_pct, name="% High Risk",
                             mode="lines+markers",
                             line=dict(color="#e74c3c", width=2, dash="dot")))
    fig.update_layout(title="Class Risk Trend Over Time",
                      xaxis_title="Week", yaxis_title="Risk (%)",
                      height=320, margin=dict(t=50, b=30))
    return fig

def plot_feature_box(result_df, feature):
    fig = px.box(
        result_df, x="risk_level", y=feature, color="risk_level",
        color_discrete_map={
            "High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#27ae60"
        },
        title=f"{feature} by Risk Level",
        category_orders={"risk_level": ["Low Risk", "Medium Risk", "High Risk"]},
    )
    fig.update_layout(height=350, showlegend=False, margin=dict(t=50, b=30))
    return fig


# ---------------------------------------------------------------
# ALERT PANEL — teacher fills in contact details and sends
# ---------------------------------------------------------------
def show_alert_panel(
    student_idx:  int,
    result_df:    pd.DataFrame,
    explanation:  dict | None,
    teacher_name: str,
):
    """
    UI panel where teacher fills in contact details
    and sends SMS + Email alerts to student and parent.
    """
    st.markdown(
        '<div class="section-header">📲 Send Alert to Student & Parent</div>',
        unsafe_allow_html=True
    )

    prob      = result_df.loc[student_idx, "dropout_probability"]
    risk      = result_df.loc[student_idx, "risk_level"]
    urgency   = result_df.loc[student_idx, "counseling_urgency"]
    top_factors = explanation["top_risk_factors"] if explanation else []

    st.markdown(f"""
    <div class="send-alert-box">
        <h4>📨 Alert Details</h4>
        <p>You are about to send an alert for
        <strong>Student {student_idx:04d}</strong> —
        Risk Level: <strong style="color:#e74c3c;">{risk}</strong>
        ({round(prob*100,1)}%)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 👤 Student Contact Info")
    col1, col2 = st.columns(2)
    with col1:
        student_name  = st.text_input("Student Name",  placeholder="e.g. Rahul Sharma")
        student_phone = st.text_input("Student Phone", placeholder="e.g. +919876543210")
    with col2:
        student_email = st.text_input("Student Email", placeholder="e.g. rahul@email.com")

    st.markdown("#### 👨‍👩‍👦 Parent Contact Info")
    col3, col4 = st.columns(2)
    with col3:
        parent_name  = st.text_input("Parent Name",  placeholder="e.g. Mr. Sharma")
        parent_phone = st.text_input("Parent Phone", placeholder="e.g. +919876543210")
    with col4:
        parent_email = st.text_input("Parent Email", placeholder="e.g. parent@email.com")

    st.markdown("#### ✏️ Custom Message (Optional)")
    custom_msg = st.text_area(
        "Add a personal note from teacher",
        placeholder="e.g. Please come to my office this week to discuss your progress.",
        height=100
    )

    st.markdown("#### ⚙️ Alert Channels")
    col5, col6 = st.columns(2)
    with col5:
        send_sms   = st.checkbox("📱 Send SMS (Twilio)",  value=True)
    with col6:
        send_email = st.checkbox("📧 Send Email (Gmail)", value=True)

    st.markdown("---")

    # Send button
    if st.button("🚀 Send Alert Now", type="primary", use_container_width=True):

        # Validation
        if not student_name:
            st.error("❌ Please enter the student name.")
            return

        if not any([student_phone, student_email, parent_phone, parent_email]):
            st.error("❌ Please enter at least one contact detail.")
            return

        with st.spinner("Sending alerts..."):
            alert_system = AlertSystem()

            results = alert_system.send_alert(
                student_name   = student_name,
                student_phone  = student_phone,
                student_email  = student_email,
                parent_name    = parent_name  or "Parent/Guardian",
                parent_phone   = parent_phone,
                parent_email   = parent_email,
                risk_level     = risk,
                risk_score     = prob,
                urgency        = urgency,
                top_factors    = top_factors,
                teacher_name   = teacher_name,
                custom_message = custom_msg,
                send_sms       = send_sms,
                send_email     = send_email,
            )

        # Show results
        st.markdown("#### 📋 Alert Results")
        r1, r2, r3, r4 = st.columns(4)

        def show_result(col, label, result):
            with col:
                if result is None:
                    st.info(f"{label}\nSkipped")
                elif result.get("success"):
                    st.success(f"✅ {label}\nSent!")
                else:
                    st.error(f"❌ {label}\n{result.get('error','Failed')}")

        show_result(r1, "📱 Student SMS",   results["student_sms"])
        show_result(r2, "📱 Parent SMS",    results["parent_sms"])
        show_result(r3, "📧 Student Email", results["student_email"])
        show_result(r4, "📧 Parent Email",  results["parent_email"])

        # Log the alert
        logger.info(
            f"Teacher {teacher_name} sent alert for Student {student_idx:04d} "
            f"| Risk: {risk} | Results: {results}"
        )


# ---------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------
def main():

    # ── Header ──────────────────────────────────────────────────
    st.markdown(
        '<div class="main-header">📊 Teacher Dashboard — Class Risk Monitor</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#7f8c8d;'>"
        "Monitor student dropout risk and send alerts to students and parents</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Load Data ────────────────────────────────────────────────
    result_df    = load_and_classify()
    risk_history = load_risk_history()
    alerts_df    = load_alerts()

    # ── Sidebar ──────────────────────────────────────────────────
    st.sidebar.title("📊 Teacher Controls")
    st.sidebar.markdown("---")

    teacher_name = st.sidebar.text_input(
        "Your Name (Teacher)", placeholder="e.g. Prof. Sharma"
    )

    selected_risks = st.sidebar.multiselect(
        "Filter by Risk Level",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"],
    )

    prob_threshold = st.sidebar.slider(
        "Min Risk Score (%)", min_value=0, max_value=100, value=0, step=5
    )

    numeric_cols     = result_df.select_dtypes(include="number").columns.tolist()
    numeric_cols     = [c for c in numeric_cols if c != "dropout_probability"]
    selected_feature = st.sidebar.selectbox("Feature to Explore", numeric_cols[:10])

    st.sidebar.markdown("---")
    st.sidebar.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Apply filters
    filtered_df = result_df[
        (result_df["risk_level"].isin(selected_risks)) &
        (result_df["dropout_probability"] * 100 >= prob_threshold)
    ]

    # ── Row 1: Metric Cards ──────────────────────────────────────
    total  = len(result_df)
    high   = (result_df["risk_level"] == "High Risk").sum()
    medium = (result_df["risk_level"] == "Medium Risk").sum()
    low    = (result_df["risk_level"] == "Low Risk").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card card-blue">
            <h2>👥 {total}</h2><p>Total Enrolled</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card card-red">
            <h2>🔴 {high}</h2><p>High Risk</p>
            <small>{round(high/total*100,1)}%</small></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card card-yellow">
            <h2>🟡 {medium}</h2><p>Medium Risk</p>
            <small>{round(medium/total*100,1)}%</small></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card card-green">
            <h2>🟢 {low}</h2><p>Low Risk</p>
            <small>{round(low/total*100,1)}%</small></div>""",
            unsafe_allow_html=True)

    st.divider()

    # ── Row 2: Charts ────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Class Overview</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_donut(result_df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_histogram(result_df), use_container_width=True)

    # ── Row 3: Trend ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Weekly Risk Trend</div>',
                unsafe_allow_html=True)
    trend_fig = plot_trend(risk_history)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("📭 Run Early Warning System to generate trend data.")

    # ── Row 4: Alerts ────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Active Alerts</div>',
                unsafe_allow_html=True)
    if not alerts_df.empty:
        st.error(f"⚠️ {len(alerts_df)} students need attention!")
        for _, alert in alerts_df.head(10).iterrows():
            css = "alert-box alert-rising" \
                if alert.get("alert_type") == "RISING RISK" else "alert-box"
            st.markdown(f"""
            <div class="{css}">
                <strong>[{alert.get('alert_type','ALERT')}]</strong>
                {alert.get('student_id','?')} —
                {alert.get('message','')}
                <span style='float:right; color:#7f8c8d;'>
                    {alert.get('timestamp','')}
                </span>
            </div>""", unsafe_allow_html=True)
        if len(alerts_df) > 10:
            st.download_button("📥 Download All Alerts",
                               data=alerts_df.to_csv(index=False),
                               file_name="alerts.csv", mime="text/csv")
    else:
        st.success("✅ No active alerts.")

    st.divider()

    # ── Row 5: Feature Explorer ──────────────────────────────────
    st.markdown('<div class="section-header">🔬 Feature Explorer</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        plot_feature_box(result_df, selected_feature),
        use_container_width=True
    )

    st.divider()

    # ── Row 6: Student Table ─────────────────────────────────────
    st.markdown(
        f'<div class="section-header">📋 Student Risk Table ({len(filtered_df)})</div>',
        unsafe_allow_html=True
    )
    display_cols = [c for c in [
        "dropout_probability", "risk_level", "risk_emoji", "counseling_urgency",
        "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
    ] if c in filtered_df.columns]

    table_df = filtered_df[display_cols].copy().sort_values(
        "dropout_probability", ascending=False
    )
    table_df["dropout_probability"] = (
        table_df["dropout_probability"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(table_df, use_container_width=True, height=400)
    st.download_button("📥 Download Risk Report",
                       data=filtered_df.to_csv(index=False),
                       file_name=f"risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

    st.divider()

    # ── Row 7: Individual Drilldown + Alert ──────────────────────
    st.markdown('<div class="section-header">🔍 Student Drilldown & Alert System</div>',
                unsafe_allow_html=True)

    high_risk_df = result_df[result_df["risk_level"] == "High Risk"]

    if not high_risk_df.empty:
        student_idx = st.selectbox(
            "Select a High Risk Student",
            options=high_risk_df.index.tolist(),
            format_func=lambda x: (
                f"Student {x:04d} — "
                f"{round(result_df.loc[x,'dropout_probability']*100,1)}% risk"
            ),
        )

        student_row = result_df.iloc[[student_idx]]
        explanation = None

        # ── SHAP Explanation ─────────────────────────────────────
        col_info, col_shap = st.columns([1, 2])

        with col_info:
            prob    = result_df.loc[student_idx, "dropout_probability"]
            level   = result_df.loc[student_idx, "risk_level"]
            emoji   = result_df.loc[student_idx, "risk_emoji"]
            urgency = result_df.loc[student_idx, "counseling_urgency"]
            st.metric("Risk Score", f"{round(prob*100,1)}%")
            st.metric("Risk Level", f"{emoji} {level}")
            st.warning(f"📋 {urgency}")

        with col_shap:
            try:
                DropoutExplainer = load_explainer()
                train_df  = load_train_data()
                explainer = DropoutExplainer()
                explainer.build_explainer(train_df)
                explanation = explainer.explain_student(student_row)

                col_r, col_p = st.columns(2)
                with col_r:
                    st.markdown("**🔴 Risk Factors**")
                    for item in explanation["top_risk_factors"][:4]:
                        st.error(
                            f"📌 **{item['feature']}**  \n"
                            f"Impact: `{item['shap_value']:+.4f}`"
                        )
                with col_p:
                    st.markdown("**🔵 Protective Factors**")
                    for item in explanation["top_protective"][:4]:
                        st.success(
                            f"✅ **{item['feature']}**  \n"
                            f"Impact: `{item['shap_value']:+.4f}`"
                        )
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

        st.divider()

        # ── Counseling Recommendations ───────────────────────────
        st.markdown(
            '<div class="section-header">🎯 Personalized Counseling Plan</div>',
            unsafe_allow_html=True
        )

        try:
            if explanation:
                import importlib.util as ilu
                rec_path = os.path.join(PROJECT_ROOT, "counseling", "recommender.py")
                spec     = ilu.spec_from_file_location("recommender", rec_path)
                mod      = ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)

                recommender     = mod.CounselingRecommender()
                prob_val        = result_df.loc[student_idx, "dropout_probability"]
                risk_val        = result_df.loc[student_idx, "risk_level"]
                recommendations = recommender.get_recommendations(
                    explanation, risk_val, prob_val
                )

                # Summary + Priority Action
                st.info(f"📋 {recommendations['summary']}")

                if recommendations["priority_action"]:
                    pa = recommendations["priority_action"]
                    st.error(
                        f"⚡ **Priority Action:** {pa.get('action','')}  \n"
                        f"📍 **Resource:** {pa.get('resource','')}"
                    )

                # Expandable cards for each recommendation
                col_r, col_g = st.columns(2)
                with col_r:
                    st.markdown("**📌 Personalized Plan:**")
                    for rec in recommendations["personalized_recommendations"]:
                        with st.expander(
                            f"{rec['icon']} {rec['issue']} [{rec['priority']}]"
                        ):
                            st.markdown(f"**Action:** {rec['action']}")
                            st.markdown(f"**Resource:** {rec['resource']}")
                            st.markdown(f"**Timeline:** {rec['timeline']}")

                with col_g:
                    st.markdown("**📋 General Actions:**")
                    for rec in recommendations["general_recommendations"]:
                        st.success(
                            f"{rec['icon']} {rec['action']}  \n"
                            f"→ *{rec['resource']}*"
                        )
            else:
                st.info("SHAP explanation needed for counseling recommendations.")

        except Exception as e:
            st.warning(f"Recommendations unavailable: {e}")

        st.divider()

        # ── ALERT PANEL ──────────────────────────────────────────
        show_alert_panel(
            student_idx  = student_idx,
            result_df    = result_df,
            explanation  = explanation,
            teacher_name = teacher_name or "Teacher",
        )

    else:
        st.info("No high risk students with current filters.")

    # ── Footer ───────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#bdc3c7; font-size:0.85rem;'>"
        "🔒 Confidential — Authorized Faculty Only | "
        "Powered by Random Forest + SHAP + Twilio</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()