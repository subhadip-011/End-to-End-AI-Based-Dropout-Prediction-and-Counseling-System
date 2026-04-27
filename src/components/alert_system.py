import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
# HOW THIS WORKS
#
# Teacher fills in student/parent contact details in dashboard
# Clicks "Send Alert" button
# System sends:
#   1. SMS via Twilio   → to student phone + parent phone
#   2. Email via Gmail  → to student email + parent email
#
# Credentials are stored in .env file — NEVER hardcoded
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# CONFIG — reads from environment variables
# ---------------------------------------------------------------
@dataclass
class AlertConfig:
    # Twilio credentials (from .env)
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token:  str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_from_number: str = os.getenv("TWILIO_FROM_NUMBER", "")

    # Gmail credentials (from .env)
    gmail_address:  str = os.getenv("GMAIL_ADDRESS", "")
    gmail_password: str = os.getenv("GMAIL_APP_PASSWORD", "")  # App password not main password


# ---------------------------------------------------------------
# SMS SENDER — Twilio
# ---------------------------------------------------------------
class SMSSender:
    def __init__(self):
        self.config = AlertConfig()

    def send(self, to_number: str, message: str, recipient_name: str) -> dict:
        """
        Sends an SMS using Twilio API.

        WHY Twilio?
        Industry standard for SMS — used by Uber, Airbnb, WhatsApp.
        Simple REST API, reliable delivery, detailed logs.

        Parameters:
        - to_number      : recipient phone number (e.g. +919876543210)
        - message        : SMS body text
        - recipient_name : for logging only (student/parent)
        """
        try:
            from twilio.rest import Client

            # Validate credentials exist
            if not self.config.twilio_account_sid:
                return {
                    "success": False,
                    "error": "Twilio credentials not configured in .env file"
                }

            client = Client(
                self.config.twilio_account_sid,
                self.config.twilio_auth_token
            )

            sms = client.messages.create(
                body=message,
                from_=self.config.twilio_from_number,
                to=to_number
            )

            logger.info(f"SMS sent to {recipient_name} ({to_number}) | SID: {sms.sid}")
            return {"success": True, "sid": sms.sid}

        except Exception as e:
            logger.error(f"SMS failed to {recipient_name}: {e}")
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------
# EMAIL SENDER — Gmail SMTP
# ---------------------------------------------------------------
class EmailSender:
    def __init__(self):
        self.config = AlertConfig()

    def send(self, to_email: str, subject: str,
             body: str, recipient_name: str) -> dict:
        """
        Sends an email using Gmail SMTP.

        WHY Gmail SMTP?
        Free, reliable, no external service needed.
        Uses App Password (not main Gmail password) for security.

        HOW TO GET APP PASSWORD:
        Gmail → Settings → Security → 2-Step Verification → App Passwords
        Generate one for 'Mail' and put it in .env as GMAIL_APP_PASSWORD
        """
        try:
            if not self.config.gmail_address:
                return {
                    "success": False,
                    "error": "Gmail credentials not configured in .env file"
                }

            # Build email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = self.config.gmail_address
            msg["To"]      = to_email

            # HTML email body — looks professional
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
                <div style="background:#e74c3c; padding:20px; border-radius:8px 8px 0 0;">
                    <h2 style="color:white; margin:0;">⚠️ Academic Alert</h2>
                    <p style="color:#fdecea; margin:5px 0 0 0;">
                        Dropout Prediction & Counseling System
                    </p>
                </div>
                <div style="background:#f8f9fa; padding:25px; border-radius:0 0 8px 8px;
                            border:1px solid #dee2e6;">
                    {body.replace(chr(10), '<br>')}
                    <hr style="border:none; border-top:1px solid #dee2e6; margin:20px 0;">
                    <p style="color:#7f8c8d; font-size:0.85rem;">
                        This is an automated alert from the Student Dropout Prediction System.
                        Please contact the institution for more information.
                    </p>
                </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, "html"))

            # Send via Gmail SMTP
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(
                    self.config.gmail_address,
                    self.config.gmail_password
                )
                server.sendmail(
                    self.config.gmail_address,
                    to_email,
                    msg.as_string()
                )

            logger.info(f"Email sent to {recipient_name} ({to_email})")
            return {"success": True}

        except Exception as e:
            logger.error(f"Email failed to {recipient_name}: {e}")
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------
# ALERT COMPOSER — builds message content
# ---------------------------------------------------------------
class AlertComposer:
    """
    Builds the actual message text for SMS and Email.
    Separated from the sender so content can be
    changed without touching sending logic.
    """

    @staticmethod
    def build_student_sms(
        student_name: str,
        risk_level: str,
        risk_score: float,
        top_factors: list,
        teacher_name: str
    ) -> str:
        factors_text = ", ".join([f["feature"] for f in top_factors[:2]])
        return (
            f"Dear {student_name},\n\n"
            f"Your teacher {teacher_name} has flagged your academic status.\n"
            f"Risk Level: {risk_level} ({round(risk_score*100, 1)}%)\n"
            f"Key concern: {factors_text}\n\n"
            f"Please schedule a counseling session at your earliest convenience.\n"
            f"- Dropout Prevention System"
        )

    @staticmethod
    def build_parent_sms(
        student_name: str,
        parent_name: str,
        risk_level: str,
        risk_score: float,
        teacher_name: str
    ) -> str:
        return (
            f"Dear {parent_name},\n\n"
            f"This is an academic alert regarding {student_name}.\n"
            f"Risk Level: {risk_level} ({round(risk_score*100, 1)}%)\n"
            f"Teacher {teacher_name} recommends an urgent counseling session.\n\n"
            f"Please contact the institution immediately.\n"
            f"- Dropout Prevention System"
        )

    @staticmethod
    def build_student_email(
        student_name: str,
        risk_level: str,
        risk_score: float,
        top_factors: list,
        urgency: str,
        teacher_name: str,
        custom_message: str = ""
    ) -> tuple:
        subject = f"⚠️ Academic Risk Alert — Action Required"

        factors_html = "\n".join([
            f"• <strong>{f['feature']}</strong> (impact: {f['shap_value']:+.3f})"
            for f in top_factors[:3]
        ])

        body = f"""
Dear {student_name},

Your teacher <strong>{teacher_name}</strong> has reviewed your academic profile
and identified the following concern:

<div style="background:#fdecea; padding:15px; border-radius:8px; margin:15px 0;">
    <strong>Risk Level:</strong> {risk_level}<br>
    <strong>Risk Score:</strong> {round(risk_score*100, 1)}%<br>
    <strong>Recommended Action:</strong> {urgency}
</div>

<strong>Key factors contributing to your risk:</strong>
{factors_html}

{f'<strong>Message from your teacher:</strong><br>{custom_message}' if custom_message else ''}

Please reach out to your academic counselor or respond to this email
to schedule a session. Early intervention significantly improves outcomes.
        """
        return subject, body

    @staticmethod
    def build_parent_email(
        student_name: str,
        parent_name: str,
        risk_level: str,
        risk_score: float,
        urgency: str,
        teacher_name: str,
        custom_message: str = ""
    ) -> tuple:
        subject = f"⚠️ Academic Alert: {student_name} Needs Attention"

        body = f"""
Dear {parent_name},

We are reaching out regarding your child <strong>{student_name}</strong>.

Our academic monitoring system, reviewed by <strong>{teacher_name}</strong>,
has identified a concern:

<div style="background:#fdecea; padding:15px; border-radius:8px; margin:15px 0;">
    <strong>Risk Level:</strong> {risk_level}<br>
    <strong>Risk Score:</strong> {round(risk_score*100, 1)}%<br>
    <strong>Recommended Action:</strong> {urgency}
</div>

{f'<strong>Note from teacher:</strong><br>{custom_message}' if custom_message else ''}

We encourage you to contact the institution to discuss next steps.
Early support can make a significant difference in your child's academic journey.
        """
        return subject, body


# ---------------------------------------------------------------
# MAIN ALERT SYSTEM — orchestrates everything
# ---------------------------------------------------------------
class AlertSystem:
    def __init__(self):
        self.sms_sender   = SMSSender()
        self.email_sender = EmailSender()
        self.composer     = AlertComposer()

    def send_alert(
        self,
        # Student info
        student_name:   str,
        student_phone:  str,
        student_email:  str,
        # Parent info
        parent_name:    str,
        parent_phone:   str,
        parent_email:   str,
        # Risk info
        risk_level:     str,
        risk_score:     float,
        urgency:        str,
        top_factors:    list,
        # Teacher info
        teacher_name:   str,
        custom_message: str = "",
        # Options
        send_sms:       bool = True,
        send_email:     bool = True,
    ) -> dict:
        """
        Master method — sends all alerts in one call.
        Returns a results dictionary with success/failure for each channel.
        """
        logger.info(f"Sending alerts for {student_name} | Risk: {risk_level}")

        results = {
            "student_sms"  : None,
            "parent_sms"   : None,
            "student_email": None,
            "parent_email" : None,
        }

        # ── SMS Alerts ──────────────────────────────────────────
        if send_sms:
            # Student SMS
            if student_phone:
                msg = self.composer.build_student_sms(
                    student_name, risk_level, risk_score,
                    top_factors, teacher_name
                )
                results["student_sms"] = self.sms_sender.send(
                    student_phone, msg, f"{student_name} (student)"
                )

            # Parent SMS
            if parent_phone:
                msg = self.composer.build_parent_sms(
                    student_name, parent_name, risk_level,
                    risk_score, teacher_name
                )
                results["parent_sms"] = self.sms_sender.send(
                    parent_phone, msg, f"{parent_name} (parent)"
                )

        # ── Email Alerts ────────────────────────────────────────
        if send_email:
            # Student Email
            if student_email:
                subject, body = self.composer.build_student_email(
                    student_name, risk_level, risk_score,
                    top_factors, urgency, teacher_name, custom_message
                )
                results["student_email"] = self.email_sender.send(
                    student_email, subject, body,
                    f"{student_name} (student)"
                )

            # Parent Email
            if parent_email:
                subject, body = self.composer.build_parent_email(
                    student_name, parent_name, risk_level,
                    risk_score, urgency, teacher_name, custom_message
                )
                results["parent_email"] = self.email_sender.send(
                    parent_email, subject, body,
                    f"{parent_name} (parent)"
                )

        logger.info(f"Alert results for {student_name}: {results}")
        return results