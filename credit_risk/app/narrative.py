"""Module 6: LLM narrative engine using Anthropic Claude API."""

import os
from datetime import datetime, timezone

from anthropic import Anthropic

SYSTEM_PROMPT = (
    "You are a senior credit analyst at an Indian fintech lender. "
    "You write precise, professional credit assessment memos. "
    "Your tone is factual and empathetic — you explain risk clearly "
    "without being judgmental about the borrower. You always explain "
    "WHY a risk factor matters, not just WHAT it is. "
    "You never use generic filler phrases like 'it is worth noting' "
    "or 'in conclusion'. Write in plain English, no jargon."
)

RECOMMENDATIONS = [
    "APPROVE WITH CONDITIONS",
    "REFER FOR REVIEW",
    "APPROVE",
    "DECLINE",
]


def format_drivers(drivers: list) -> str:
    lines = []
    for driver in drivers:
        lines.append(
            f"• {driver['human_readable']}: SHAP impact {driver['shap_value']:+.3f}"
        )
    return "\n".join(lines) if lines else "• None identified"


def format_groups(groups: dict) -> str:
    sorted_groups = sorted(groups.items(), key=lambda item: abs(item[1]), reverse=True)
    return "\n".join(f"• {group}: {contribution:+.3f}" for group, contribution in sorted_groups)


def thin_file_section(explanation: dict) -> str:
    if not explanation.get("is_thin_file"):
        return ""
    tf = explanation.get("thin_file_explanation") or {}
    return (
        "THIN-FILE ASSESSMENT CONTEXT:\n"
        f"• Alternate credit score: {tf.get('alt_credit_score', 'N/A')}\n"
        f"• Data confidence: {tf.get('thin_file_confidence', 0):.0%}\n"
        f"• Alt score SHAP impact: {tf.get('alt_score_shap', 0):+.3f}\n"
        f"• Note: {tf.get('note', '')}"
    )


def format_counterfactuals(counterfactuals: list | None) -> str:
    if not counterfactuals:
        return "• No actionable improvements identified"
    lines = []
    for cf in counterfactuals:
        reduction = cf.get("estimated_probability_reduction")
        if reduction is not None:
            lines.append(
                f"• {cf['action']}: could reduce default probability by {reduction:.1%}"
            )
        else:
            note = cf.get("note", cf.get("action", "No improvement identified"))
            lines.append(f"• {note}")
    return "\n".join(lines)


def _build_user_prompt(explanation: dict) -> str:
    probs = explanation["class_probabilities"]
    return f"""
Write a credit assessment memo for this loan applicant.

APPLICANT PROFILE:
- Applicant ID: {explanation['applicant_id']}
- Thin-file borrower: {explanation['is_thin_file']}
- Predicted default probability: {explanation['predicted_default_probability']:.1%}
- Risk classification: {explanation['predicted_risk_label']}
- Risk confidence: Low {probs['Low']:.1%} |
                   Medium {probs['Medium']:.1%} |
                   High {probs['High']:.1%}

TOP RISK FACTORS (these INCREASE default risk):
{format_drivers(explanation['top_risk_drivers'])}

TOP PROTECTIVE FACTORS (these DECREASE default risk):
{format_drivers(explanation['top_risk_mitigants'])}

RISK BREAKDOWN BY CATEGORY:
{format_groups(explanation['feature_group_contributions'])}

{thin_file_section(explanation)}

RECOMMENDED ACTIONS FOR BORROWER:
{format_counterfactuals(explanation.get('counterfactuals'))}

Write the memo in these exact sections:
1. CREDIT SUMMARY (2-3 sentences: risk verdict, primary reason, confidence)
2. KEY RISK FACTORS (bullet each driver with a one-sentence explanation of
   why it matters for repayment — not just what it is)
3. MITIGATING STRENGTHS (bullet each mitigant similarly)
4. THIN-FILE ASSESSMENT (only if is_thin_file=True: explain how alternate
   data was used and what confidence level it provides)
5. RECOMMENDED ACTIONS (3 specific, actionable steps the borrower can take
   with estimated impact — use the counterfactuals above)
6. ANALYST RECOMMENDATION: one of APPROVE / APPROVE WITH CONDITIONS /
   REFER FOR REVIEW / DECLINE — with a one-line rationale

Keep the total memo under 400 words. Be specific with numbers.
"""


def _parse_recommendation(narrative: str) -> str:
    last_section = narrative[-500:].upper()
    for rec in RECOMMENDATIONS:
        if rec in last_section:
            return rec
    return "REFER FOR REVIEW"


def _mock_narrative(applicant_id: str) -> dict:
    return {
        "applicant_id": applicant_id,
        "narrative": (
            "API key not configured. Set ANTHROPIC_API_KEY in .env to "
            "generate AI narratives. This is a demo placeholder."
        ),
        "recommendation": "REFER FOR REVIEW",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def generate_credit_narrative(explanation: dict) -> dict:
    applicant_id = explanation["applicant_id"]
    generated_at = datetime.now(timezone.utc).isoformat()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _mock_narrative(applicant_id)

    client = Anthropic()
    user_prompt = _build_user_prompt(explanation)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    narrative_text = response.content[0].text
    recommendation = _parse_recommendation(narrative_text)

    return {
        "applicant_id": applicant_id,
        "narrative": narrative_text,
        "recommendation": recommendation,
        "generated_at": generated_at,
    }
