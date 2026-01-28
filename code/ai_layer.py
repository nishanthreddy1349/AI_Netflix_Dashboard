import json
import streamlit as st
from openai import OpenAI

def _get_client():
    return OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        project=st.secrets["OPENAI_PROJECT_ID"],
    )
st.write("Project ID loaded:", st.secrets.get("OPENAI_PROJECT_ID"))

def _call_with_fallback(client: OpenAI, models: list[str], messages: list[dict], temperature: float = 0.2) -> tuple[str, str]:
    """
    Try models in order. Returns: (text_output, model_used)
    """
    last_err = None
    for m in models:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
            )
            text = resp.choices[0].message.content
            return text, m
        except Exception as e:
            last_err = e
            continue
    raise last_err

def generate_dashboard_summary(evidence: dict, model: str = "gpt-4o-mini", fallback_models: list[str] | None = None) -> dict:
    client = _get_client()

    # Try selected model first, then fallbacks
    models = [model] + (fallback_models or [])

    system_msg = (
        "You are an analytics assistant. Summarize the dashboard using ONLY the provided evidence. "
        "Be specific with numbers (percent changes, totals, top titles). "
        "Return JSON only with keys: headline, summary_bullets, key_changes, next_checks."
    )

    user_payload = {
        "evidence": evidence,
        "rules": [
            "Do not invent numbers. Use only provided evidence.",
            "summary_bullets: 3-5 bullets",
            "key_changes: 2-4 bullets (use % changes if available)",
            "next_checks: 2-4 bullets phrased as analytics breakdowns (by device/country/title), avoid generic marketing advice",
        ],
        "output_format": {
            "headline": "string",
            "summary_bullets": ["..."],
            "key_changes": ["..."],
            "next_checks": ["..."],
        },
    }

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    text, model_used = _call_with_fallback(client, models, messages, temperature=0.2)

    try:
        out = json.loads(text)
    except Exception:
        out = {
            "headline": "AI Summary",
            "summary_bullets": [text],
            "key_changes": [],
            "next_checks": [],
        }

    out["_model_used"] = model_used
    return out

def explain_change(evidence: dict, metric: str = "total_watch_minutes",
                   model: str = "gpt-4o-mini", fallback_models: list[str] | None = None) -> dict:
    client = _get_client()
    models = [model] + (fallback_models or [])

    system_msg = (
        "You are an analytics assistant. Explain WHY the selected metric changed using ONLY the evidence. "
        "Separate facts from hypotheses. Avoid unsupported causality. "
        "Return JSON only with keys: headline, what_changed, likely_drivers, next_checks."
    )

    user_payload = {
        "metric_to_explain": metric,
        "evidence": evidence,
        "rules": [
            "Do not invent numbers. Use only provided evidence.",
            "what_changed: facts with numbers (2-4 bullets).",
            "likely_drivers: grounded drivers referencing device/country/title (3-6 bullets).",
            "If you must speculate, label it as 'Hypothesis:' and keep it minimal.",
            "next_checks: 2-4 bullets phrased as analytics breakdowns (segment/compare), not generic advice.",
        ],
        "output_format": {
            "headline": "string",
            "what_changed": ["..."],
            "likely_drivers": ["..."],
            "next_checks": ["..."],
        },
    }

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    text, model_used = _call_with_fallback(client, models, messages, temperature=0.2)

    try:
        out = json.loads(text)
    except Exception:
        out = {
            "headline": "AI Explanation",
            "what_changed": [text],
            "likely_drivers": [],
            "next_checks": [],
        }

    out["_model_used"] = model_used
    return out
    


