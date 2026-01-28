from pathlib import Path
import streamlit as st
import plotly.express as px

from data_layer import load_data
from evidence_builder import build_evidence
from ai_layer import generate_dashboard_summary, explain_change
from feedback_store import init_db, save_review, get_recent_reviews


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Streaming Analytics + AI Insights", layout="wide")


# -----------------------------
# Netflix-style CSS (cards, chips, spacing)
# -----------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      .hero {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        background: rgba(255,255,255,0.02);
      }
      .subtle { color: rgba(255,255,255,0.70); }
      .pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        font-size: 0.85rem;
        margin-right: 6px;
        margin-top: 6px;
      }
      .section-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 12px;
      }
      .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 10px;
      }
      .muted {
        color: rgba(255,255,255,0.70);
        font-size: 0.92rem;
      }
      .chip {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 8px;
      }
      .list-tight ul { margin-top: 6px; margin-bottom: 0px; padding-left: 1.2rem; }
      .list-tight li { margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers for UI cards
# -----------------------------
def render_card(title: str, subtitle: str | None = None):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="muted">{subtitle}</div>', unsafe_allow_html=True)


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def render_bullets(items: list[str]):
    if not items:
        return
    html = "<div class='list-tight'><ul>"
    for x in items:
        html += f"<li>{x}</li>"
    html += "</ul></div>"
    st.markdown(html, unsafe_allow_html=True)


def fmt_delta(pct_val):
    if pct_val is None:
        return None
    return f"{pct_val:.2f}%"


# -----------------------------
# Load data (cached)
# -----------------------------
@st.cache_data
def get_final_df():
    return load_data()


final_df = get_final_df()


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

min_date = final_df["watch_date"].min().date()
max_date = final_df["watch_date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

genre_options = ["All"] + sorted(final_df["genre_primary"].dropna().unique().tolist())
selected_genre = st.sidebar.selectbox("Select genre", genre_options)

filtered_df = final_df[
    (final_df["watch_date"].dt.date >= start_date)
    & (final_df["watch_date"].dt.date <= end_date)
].copy()

if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["genre_primary"] == selected_genre]


# -----------------------------
# Phase 1 metrics
# -----------------------------
filtered_df["watch_day"] = filtered_df["watch_date"].dt.date

watch_time_trend = (
    filtered_df.groupby("watch_day", as_index=False)["watch_duration_minutes"].sum()
)

watch_time_by_genre = (
    filtered_df.groupby("genre_primary", as_index=False)["watch_duration_minutes"]
    .sum()
    .sort_values("watch_duration_minutes", ascending=False)
)

top_titles = (
    filtered_df.groupby("title", as_index=False)["watch_duration_minutes"]
    .sum()
    .sort_values("watch_duration_minutes", ascending=False)
    .head(10)
)

# Evidence (Phase 2)
evidence = build_evidence(final_df, filtered_df, start_date, end_date, selected_genre)
changes = evidence.get("changes", {})
cur = evidence.get("current_period", {})
prev = evidence.get("previous_period", {})


# Header (logo + title)

assets_dir = Path(__file__).resolve().parent.parent / "assets"
logo_path = assets_dir / "netflix_logo.png"

if logo_path.exists():
    st.image(str(logo_path), width=56)

left, right = st.columns([1, 6], vertical_alignment="center")
with left:
    if logo_path.exists():
        st.image(str(logo_path), width=56)
    else:
        st.markdown("<span class='pill'>N</span>", unsafe_allow_html=True)

with right:
    st.markdown(
        f"""
        <div class="hero">
          <div style="font-size:1.6rem; font-weight:900;">Streaming Analytics + AI Insights</div>
          <div class="subtle">Netflix-style dashboard + an AI intelligence layer that summarizes and explains changes.</div>
          <div style="margin-top:8px;">
            <span class="pill">Phase 1: Analytics</span>
            <span class="pill">Phase 2: AI Summaries</span>
            <span class="pill">Filters: {start_date} → {end_date} | Genre: {selected_genre}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Insights", "📝 Feedback"])


# =============================
# Tab 1: Dashboard
# =============================
with tab1:
    total_watch_minutes = filtered_df["watch_duration_minutes"].sum()
    unique_users = filtered_df["user_id"].nunique()
    unique_titles = filtered_df["title"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Watch Minutes", f"{int(total_watch_minutes):,}")
    c2.metric("Active Users", f"{int(unique_users):,}")
    c3.metric("Titles Watched", f"{int(unique_titles):,}")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.line(
            watch_time_trend,
            x="watch_day",
            y="watch_duration_minutes",
            title="Watch Time Over Time",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            watch_time_by_genre,
            x="genre_primary",
            y="watch_duration_minutes",
            title="Watch Time by Genre",
        )
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        top_titles,
        x="watch_duration_minutes",
        y="title",
        orientation="h",
        title="Top Titles by Watch Time",
    ).update_layout(yaxis=dict(autorange="reversed"))

    st.plotly_chart(fig3, use_container_width=True)


# =============================
# Tab 2: AI Insights (Improved UI/UX)
# =============================
with tab2:
    # Persist outputs across reruns
    if "ai_summary" not in st.session_state:
        st.session_state.ai_summary = None
    if "ai_explain" not in st.session_state:
        st.session_state.ai_explain = None

    # --- At a glance (top) ---
    render_card("At a glance", "Current vs previous period (auto-colored deltas)")

    changes = evidence.get("changes", {})
    cur = evidence.get("current_period", {})

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Total Watch Minutes",
        f"{cur.get('total_watch_minutes', 0):,.1f}",
        fmt_delta(changes.get("total_watch_minutes_pct_change")),
    )
    m2.metric(
        "Active Users",
        f"{cur.get('active_users', 0):,}",
        fmt_delta(changes.get("active_users_pct_change")),
    )
    m3.metric(
        "Titles Watched",
        f"{cur.get('titles_watched', 0):,}",
        fmt_delta(changes.get("titles_watched_pct_change")),
    )

    pw = changes.get("previous_window")
    if pw:
        st.markdown(
            f"<span class='chip'>Prev window: {pw.get('start_date')} → {pw.get('end_date')} ({pw.get('days')} days)</span>",
            unsafe_allow_html=True,
        )

    note = evidence.get("note", "")
    if note:
        st.info(note)

    close_card()

    # Model selector (top of AI tab)
    render_card("Model", "Choose a model and compare outputs. If unavailable, auto-fallback will kick in.")

    model_choice = st.selectbox(
        "Model",
        options=[
            "gpt-5.2",
            "gpt-5.2-mini",
            "gpt-4o-mini"
        ],
        index=0,
        key="model_choice"
    )

    fallback_models = ["gpt-5.2-mini", "gpt-4o-mini"]  # tried in order if selected fails

    close_card()

    # --- Two-column layout ---
    left_col, right_col = st.columns([1, 1], gap="large")

    # ==========================
    # LEFT: AI Summary
    # ==========================
    with left_col:
        render_card("✨ AI Summary", "One-click executive summary grounded in dashboard evidence")

        run_summary = st.button("Generate Summary", key="btn_summary")

        if run_summary:
            with st.spinner("Generating AI summary..."):
                try:
                    st.session_state.ai_summary = generate_dashboard_summary(
    evidence,
    model=model_choice,
    fallback_models=fallback_models
)
                except Exception as e:
                    st.error(f"AI summary failed: {e}")

        if st.session_state.ai_summary:
            res = st.session_state.ai_summary
            st.caption(f"Model used: {res.get('_model_used', 'unknown')}")

            st.markdown(f"<span class='chip'>{res.get('headline','AI Summary')}</span>", unsafe_allow_html=True)

            st.markdown("<span class='chip'>Summary</span>", unsafe_allow_html=True)
            render_bullets(res.get("summary_bullets", []))

            if res.get("key_changes"):
                st.markdown("<span class='chip'>Key changes</span>", unsafe_allow_html=True)
                render_bullets(res.get("key_changes", []))

            if res.get("next_checks"):
                st.markdown("<span class='chip'>Next checks</span>", unsafe_allow_html=True)
                render_bullets(res.get("next_checks", []))

        close_card()

    # ==========================
    # RIGHT: Explain Change
    # ==========================
    with right_col:
        render_card("🔍 Explain Change", "Explain the change for a selected KPI (facts + drivers + next checks)")

        metric_choice = st.selectbox(
            "Metric to explain",
            options=["total_watch_minutes", "active_users", "titles_watched"],
            index=0,
            key="metric_choice_split"
        )

        run_explain = st.button("Explain Now", key="btn_explain")

        if run_explain:
            with st.spinner("Explaining change..."):
                try:
                    st.session_state.ai_explain = explain_change(
    evidence,
    metric=metric_choice,
    model=model_choice,
    fallback_models=fallback_models
)
                except Exception as e:
                    st.error(f"AI explanation failed: {e}")

        if st.session_state.ai_explain:
            res = st.session_state.ai_explain
            st.caption(f"Model used: {res.get('_model_used', 'unknown')}")

            st.markdown(f"<span class='chip'>{res.get('headline','AI Explanation')}</span>", unsafe_allow_html=True)

            st.markdown("<span class='chip'>What changed</span>", unsafe_allow_html=True)
            render_bullets(res.get("what_changed", []))

            if res.get("likely_drivers"):
                st.markdown("<span class='chip'>Likely drivers</span>", unsafe_allow_html=True)
                render_bullets(res.get("likely_drivers", []))

            if res.get("next_checks"):
                st.markdown("<span class='chip'>Next checks</span>", unsafe_allow_html=True)
                render_bullets(res.get("next_checks", []))

        close_card()

    # Evidence (optional)
    with st.expander("Show evidence used (debug)"):
        st.json(evidence)


# =============================
# Tab 3: Feedback (Reviews)
# =============================
with tab3:
    init_db()

    render_card("Leave a Review", "Help improve this project by sharing quick feedback")

    rating = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 5)
    comment = st.text_area("What did you like? What was confusing? Any suggestions?", height=120)

    submitted = st.button("Submit Review")
    if submitted:
        if not comment.strip():
            st.warning("Please enter a short comment before submitting.")
        else:
            save_review(int(rating), comment)
            st.success("Thanks! Your feedback was saved.")

    close_card()

    render_card("Recent Reviews", "Latest feedback captured in this deployment")
    rows = get_recent_reviews(8)
    if not rows:
        st.write("No reviews yet. Be the first to leave feedback!")
    else:
        for created_at, r, c in rows:
            st.write(f"**⭐ {r}/5** — {c}")
            st.caption(created_at)
            st.divider()
    close_card()