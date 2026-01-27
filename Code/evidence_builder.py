from __future__ import annotations
from datetime import timedelta
import pandas as pd


def _safe_pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / previous * 100.0


def _top_n(df: pd.DataFrame, group_col: str, value_col: str, n: int = 5) -> list[dict]:
    if group_col not in df.columns:
        return []
    return (
        df.groupby(group_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
        .to_dict(orient="records")
    )


def _period_aggregates(df: pd.DataFrame) -> dict:
    return {
        "total_watch_minutes": round(float(df["watch_duration_minutes"].sum()), 2),
        "active_users": int(df["user_id"].nunique()),
        "titles_watched": int(df["title"].nunique()),
        "top_genres": _top_n(df, "genre_primary", "watch_duration_minutes", 5),
        "top_titles": _top_n(df, "title", "watch_duration_minutes", 5),
        "watch_by_device": _top_n(df, "device_type", "watch_duration_minutes", 5),
        "watch_by_country": _top_n(df, "location_country", "watch_duration_minutes", 5),
    }


def _delta_table(
    cur_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    key: str,
    value: str,
    n: int = 5,
    sort_by_abs: bool = True,
) -> list[dict]:
    """
    Creates a delta table for a driver dimension.

    Output rows: {key: <name>, current: <float>, previous: <float>, delta: <float>, pct_change: <float|None>}
    Sorted by absolute delta (default) or by delta.
    """
    if key not in cur_df.columns or key not in prev_df.columns:
        return []

    cur_series = cur_df.groupby(key)[value].sum()
    prev_series = prev_df.groupby(key)[value].sum()

    combined = pd.DataFrame(
        {
            "current": cur_series,
            "previous": prev_series,
        }
    ).fillna(0.0)

    combined["delta"] = combined["current"] - combined["previous"]
    combined["pct_change"] = combined.apply(
        lambda r: _safe_pct_change(float(r["current"]), float(r["previous"])),
        axis=1,
    )

    if sort_by_abs:
        combined = combined.reindex(combined["delta"].abs().sort_values(ascending=False).index)
    else:
        combined = combined.sort_values("delta", ascending=False)

    combined = combined.head(n).reset_index().rename(columns={key: "name", "index": "name"})
    # make output key name consistent with dimension name
    combined = combined.rename(columns={"name": key})

    # Round numeric outputs for readability
    for c in ["current", "previous", "delta"]:
        combined[c] = combined[c].astype(float).round(2)

    # pct_change keep as float (rounded) or None
    combined["pct_change"] = combined["pct_change"].apply(lambda x: None if x is None else round(float(x), 2))

    return combined[[key, "current", "previous", "delta", "pct_change"]].to_dict(orient="records")


def build_evidence(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    start_date,
    end_date,
    selected_genre: str,
) -> dict:
    """
    Evidence packet for the AI layer.
    Adds driver deltas + engagement intensity so the model can answer "why" (not just read numbers).

    Option 1 behavior:
      - If previous period has no data, previous_period is empty and changes pct fields are None.
    """
    evidence = {
        "filters": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "genre": selected_genre,
        },
        "current_period": {},
        "previous_period": {},
        "changes": {},
        "note": "",
    }

    # Guard: empty current selection
    if filtered_df.empty:
        evidence["note"] = "No data available for the selected filters."
        return evidence

    # Current period aggregates
    current = _period_aggregates(filtered_df)
    evidence["current_period"] = current

    # Previous period window (same length, immediately before start_date)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    window_days = int((end_dt - start_dt).days) + 1

    prev_end_dt = start_dt - timedelta(days=1)
    prev_start_dt = prev_end_dt - timedelta(days=window_days - 1)

    prev_df = full_df[
        (full_df["watch_date"] >= prev_start_dt)
        & (full_df["watch_date"] <= prev_end_dt)
    ].copy()

    # Apply same genre filter to previous period
    if selected_genre != "All":
        prev_df = prev_df[prev_df["genre_primary"] == selected_genre]

    # Option 1: previous period unavailable
    if prev_df.empty:
        evidence["note"] = "Previous period comparison unavailable (no data in the prior time window)."
        evidence["changes"] = {
            "total_watch_minutes_pct_change": None,
            "active_users_pct_change": None,
            "titles_watched_pct_change": None,
            "previous_window": {
                "start_date": str(prev_start_dt.date()),
                "end_date": str(prev_end_dt.date()),
                "days": window_days,
            },
            "engagement": {
                "minutes_per_user_current": round(
                    current["total_watch_minutes"] / max(current["active_users"], 1), 2
                ),
                "minutes_per_user_previous": None,
                "minutes_per_user_pct_change": None,
            },
            "driver_deltas": {},
        }
        return evidence

    # Previous period aggregates
    previous = _period_aggregates(prev_df)
    evidence["previous_period"] = previous

    # % changes
    total_pct = _safe_pct_change(current["total_watch_minutes"], previous["total_watch_minutes"])
    users_pct = _safe_pct_change(float(current["active_users"]), float(previous["active_users"]))
    titles_pct = _safe_pct_change(float(current["titles_watched"]), float(previous["titles_watched"]))

    # Engagement intensity: minutes per user
    current_mpu = current["total_watch_minutes"] / max(current["active_users"], 1)
    prev_mpu = previous["total_watch_minutes"] / max(previous["active_users"], 1)
    mpu_pct = _safe_pct_change(current_mpu, prev_mpu)

    # Driver deltas: WHAT drove the change (grounded)
    driver_deltas = {
        "device_type": _delta_table(filtered_df, prev_df, "device_type", "watch_duration_minutes", n=6),
        "location_country": _delta_table(filtered_df, prev_df, "location_country", "watch_duration_minutes", n=6),
        "title": _delta_table(filtered_df, prev_df, "title", "watch_duration_minutes", n=6),
        # optional: genre breakdown (useful when genre = All)
        "genre_primary": _delta_table(filtered_df, prev_df, "genre_primary", "watch_duration_minutes", n=6),
    }

    evidence["changes"] = {
        "total_watch_minutes_pct_change": total_pct,
        "active_users_pct_change": users_pct,
        "titles_watched_pct_change": titles_pct,
        "previous_window": {
            "start_date": str(prev_start_dt.date()),
            "end_date": str(prev_end_dt.date()),
            "days": window_days,
        },
        "engagement": {
            "minutes_per_user_current": round(float(current_mpu), 2),
            "minutes_per_user_previous": round(float(prev_mpu), 2),
            "minutes_per_user_pct_change": mpu_pct,
        },
        "driver_deltas": driver_deltas,
    }

    return evidence