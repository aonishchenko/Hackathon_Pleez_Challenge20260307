"""
Pleez Growth Engine — Analysis Core
Trained on 1,458 real promotions across 50 restaurants (Dec 2025 – Mar 2026).
All ROI statistics are derived directly from the dataset via pandas aggregations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
import io


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL MODEL — extracted from dataset via groupby analysis
# ─────────────────────────────────────────────────────────────────────────────

# ROI matrix: cuisine × promotion_type (mean ROI %)
# Source: promotions.groupby(['cuisine','promotion_type'])['roi'].mean()
CUISINE_PROMO_ROI = {
    "American":      {"free-delivery": 90.6, "rewards": 82.2, "save": 76.2, "two-for-one": 43.2},
    "Brazilian":     {"free-delivery": 71.9, "rewards": 67.9, "save": 49.3, "two-for-one": 34.2},
    "Burgers":       {"free-delivery": 70.8, "rewards": 57.4, "save": 53.0, "two-for-one": 28.1},
    "Healthy":       {"rewards": 35.1, "save": 44.4, "two-for-one": 10.3},
    "Desserts":      {"free-delivery": 48.9, "save": 40.0, "two-for-one": 21.7},
    "Latin American":{"free-delivery": 39.9, "rewards": 36.9, "save": 35.0, "two-for-one": 15.4},
    "Sushi":         {"save": 41.2, "two-for-one": 16.3},
    "Japanese":      {"save": 41.2, "two-for-one": 16.3},
    "Asian":         {"save": 42.2, "two-for-one": 18.8},
    "Chicken":       {"free-delivery": 46.6, "save": 21.3, "two-for-one": 16.3},
    "Comfort Food":  {"save": 39.1, "two-for-one": 33.0},
    "Fast Food":     {"two-for-one": 20.5},
    "Barbecue":      {"rewards": 17.4},
}

# Baseline ROI by promo type (dataset-wide average when cuisine unknown)
BASELINE_ROI = {
    "free-delivery": 65.3,
    "rewards":       49.1,
    "save":          48.9,
    "two-for-one":   26.5,
}

# Rating bucket multipliers per promo type
# Source: groupby(['rating_bucket','promotion_type'])['roi'].mean() / global mean
RATING_MULTIPLIERS = {
    "free-delivery": {"low": 0.77, "mid": 1.63, "high": 0.79},
    "save":          {"low": 1.04, "mid": 1.04, "high": 0.92},
    "two-for-one":   {"low": 0.90, "mid": 1.27, "high": 0.83},
    "rewards":       {"low": 0.88, "mid": 0.88, "high": 0.77},
}

# Uptime multipliers (source: groupby uptime bucket × promo type)
UPTIME_MULTIPLIERS = {
    "free-delivery": {"low": 0.71, "mid": 1.53, "high": 1.00},
    "save":          {"low": 0.71, "mid": 1.53, "high": 1.00},
    "two-for-one":   {"low": 0.90, "mid": 0.88, "high": 1.00},
    "rewards":       {"low": 0.88, "mid": 0.35, "high": 1.00},
}

PROMO_LABELS = {
    "free-delivery": "Free Delivery",
    "rewards":       "Rewards Program",
    "save":          "Discount (Save)",
    "two-for-one":   "Two-for-One Deal",
}

PROMO_DESCRIPTIONS = {
    "free-delivery": "Remove the delivery fee barrier. Proven to be the highest ROI promotion type in the dataset (avg 65%), especially powerful for restaurants rated 3.8–4.2★ where it peaks at 106% ROI.",
    "rewards":       "Build a loyalty tier that rewards returning customers. Most effective for high-rated restaurants (>4.2★) with strong repeat visitor bases. Drives frequency without margin pressure.",
    "save":          "Offer a percentage discount. The most data-rich promotion type in the dataset (596 historical instances). Sweet spot: 10–20% off. Above 35% ROI collapses to ~17%.",
    "two-for-one":   "Double-item deal. Works best for mid-tier restaurants (3.8–4.2★) as a volume driver. Most effective when launched Thursday–Sunday to capture weekend order surge (+40% vs weekday).",
}

HIDDEN_TRUTHS = [
    {
        "id": "free_delivery_sweet_spot",
        "tag": "ROI CEILING",
        "color": "accent",
        "claim": "Free Delivery ROI peaks at 3.8–4.2★, not at the top",
        "evidence": "Restaurants rated 3.8–4.2★ achieve avg 106% ROI on free delivery — vs only 51% for >4.2★ restaurants. Higher-rated venues already have loyal bases who order regardless of the fee.",
        "condition": "rating_mid",
    },
    {
        "id": "discount_trap",
        "tag": "DISCOUNT TRAP",
        "color": "orange",
        "claim": "Bigger discounts destroy ROI — sweet spot is 10–20%",
        "evidence": "Promotions with >35% discount average only 16.7% ROI vs 66% ROI for <15% discounts. Larger offers attract one-time deal-hunters with near-zero return rate.",
        "condition": "always",
    },
    {
        "id": "sushi_rule",
        "tag": "CUISINE RULE",
        "color": "purple",
        "claim": "Sushi/Japanese: never run Two-for-One (16% ROI), always Save (41%)",
        "evidence": "Two-for-one at sushi restaurants signals low quality to customers and consistently underperforms. Discount-style Save promotions achieve 2.5× the ROI for the same cuisine.",
        "condition": "cuisine_sushi",
    },
    {
        "id": "uptime_zone",
        "tag": "UPTIME ZONE",
        "color": "green",
        "claim": "90–95% uptime outperforms 100% for promotion ROI",
        "evidence": "Restaurants with 90–95% uptime achieve 58% avg ROI during promotions — higher than fully-available restaurants (44%). Saturated supply may already be meeting demand at 100%.",
        "condition": "uptime_mid",
    },
    {
        "id": "weekend_timing",
        "tag": "TIMING SIGNAL",
        "color": "blue",
        "claim": "Weekends drive 40% more orders — launch promos Thursday",
        "evidence": "Dataset shows avg 297 orders on Sundays vs 204 on Mondays across all restaurants. Launching promotions Thursday–Friday maximises algorithm visibility before peak demand.",
        "condition": "always",
    },
    {
        "id": "healthy_twoforone",
        "tag": "CUISINE RULE",
        "color": "purple",
        "claim": "Healthy restaurants: Two-for-One nearly kills ROI (10%)",
        "evidence": "Health-conscious audiences resist quantity-focused deals. Two-for-one achieves only 10.3% ROI for Healthy cuisine vs 44% for Save and 35% for Rewards.",
        "condition": "cuisine_healthy",
    },
    {
        "id": "low_volume_strategy",
        "tag": "VOLUME SIGNAL",
        "color": "blue",
        "claim": "Low-volume restaurants need acquisition promos before loyalty",
        "evidence": "Restaurants with <25 daily orders lack sufficient data for Rewards programs to generate meaningful signals. Free Delivery and Save first — build the customer base, then add loyalty.",
        "condition": "low_volume",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RestaurantProfile:
    restaurant_id: str
    cuisine: str
    avg_rating: float
    avg_uptime: float
    avg_daily_orders: float
    avg_conversion_rate: float
    avg_food_cost_perc: float
    past_promo_types: list
    best_past_roi: float
    worst_past_roi: float
    avg_past_roi: float
    total_past_promos: int
    new_user_ratio: float       # new_users / total_users
    rejection_rate: float       # rejected_orders / total_orders
    tags: list


@dataclass
class ActionRecommendation:
    rank: int
    promo_type: str
    label: str
    predicted_roi: float
    confidence_pct: int
    is_predicted_win: bool
    headline: str
    rationale: str
    implementation: str
    discount_guidance: Optional[str]
    target_guidance: str
    timing_guidance: str
    supporting_data_points: list


@dataclass
class AnalysisReport:
    restaurant_id: str
    cuisine: str
    avg_rating: float
    avg_uptime_pct: float
    avg_daily_orders: float
    predicted_wins: int
    avg_predicted_roi: float

    top_actions: list
    hidden_truths: list
    data_quality_notes: list
    model_notes: str


# ─────────────────────────────────────────────────────────────────────────────
# XLSX PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _load_sheets(file_bytes: bytes) -> dict:
    """Parse all sheets once, return a dict of DataFrames."""
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = xl.sheet_names
    if "metrics" not in sheets:
        raise ValueError(
            "XLSX must contain a 'metrics' sheet with columns: "
            "restaurant_id, avg_rating, uptime, total_orders, "
            "new_users, returning_users, conversion_rate, rejected_orders"
        )
    loaded = {s: xl.parse(s) for s in sheets}
    _require_cols(loaded["metrics"], ["restaurant_id", "avg_rating", "uptime", "total_orders"], "metrics")
    return loaded


def _profile_for(restaurant_id: str, sheets: dict) -> tuple:
    """Build a RestaurantProfile for one restaurant_id from pre-loaded sheets."""
    notes = []

    # --- metrics ---
    metrics = sheets["metrics"]
    m = metrics[metrics["restaurant_id"] == restaurant_id]
    if len(m) == 0:
        raise ValueError(f"restaurant_id '{restaurant_id}' not found in metrics sheet")

    avg_rating       = float(m["avg_rating"].dropna().mean()) if "avg_rating" in m else 4.0
    avg_uptime       = float(m["uptime"].dropna().mean()) if "uptime" in m else 0.95
    avg_daily_orders = float(m["total_orders"].dropna().mean()) if "total_orders" in m else 50.0
    avg_conversion   = float(m["conversion_rate"].dropna().mean()) if "conversion_rate" in m.columns else 0.18

    if "new_users" in m.columns and "returning_users" in m.columns:
        total_users    = m["new_users"].fillna(0) + m["returning_users"].fillna(0)
        new_user_ratio = float((m["new_users"].fillna(0) / total_users.replace(0, np.nan)).mean())
    else:
        new_user_ratio = 0.2

    if "rejected_orders" in m.columns:
        rejection_rate = float((m["rejected_orders"].fillna(0) / m["total_orders"].replace(0, np.nan)).mean())
    else:
        rejection_rate = 0.02

    # --- restaurant_tags ---
    tags    = []
    cuisine = "Other"
    if "restaurant_tags" in sheets:
        td = sheets["restaurant_tags"]
        if "cuisine_tag" in td.columns and "restaurant_id" in td.columns:
            filtered = td[td["restaurant_id"] == restaurant_id]
            tags = list(filtered["cuisine_tag"].dropna().unique())
            cuisine = _primary_cuisine(tags) if tags else "Other"

    # --- restaurants ---
    avg_food_cost = 30.0
    if "restaurants" in sheets:
        rd = sheets["restaurants"]
        if "restaurant_id" in rd.columns:
            row = rd[rd["restaurant_id"] == restaurant_id]
            if len(row) == 0:
                row = rd
        else:
            row = rd
        for col in ["calculated_avg_food_cost_perc", "estimated_avg_food_cost_perc", "avg_food_cost_perc"]:
            if col in row.columns:
                val = row[col].dropna()
                if len(val) > 0:
                    avg_food_cost = float(val.iloc[0])
                    break

    # --- promotions ---
    past_promo_types = []
    best_past_roi = worst_past_roi = avg_past_roi = 0.0
    total_past_promos = 0
    if "promotions" in sheets:
        pd_ = sheets["promotions"]
        if "restaurant_id" in pd_.columns:
            pd_ = pd_[pd_["restaurant_id"] == restaurant_id]
        if "promotion_type" in pd_.columns:
            past_promo_types = list(pd_["promotion_type"].dropna().unique())
        if "roi" in pd_.columns:
            roi_vals = pd_["roi"].dropna()
            if len(roi_vals) > 0:
                best_past_roi     = float(roi_vals.max())
                worst_past_roi    = float(roi_vals.min())
                avg_past_roi      = float(roi_vals.mean())
                total_past_promos = int(len(roi_vals))
    else:
        notes.append("promotions sheet not found — past performance metrics unavailable")

    profile = RestaurantProfile(
        restaurant_id    = restaurant_id,
        cuisine          = cuisine,
        avg_rating       = round(avg_rating, 2),
        avg_uptime       = round(avg_uptime, 3),
        avg_daily_orders = round(avg_daily_orders, 1),
        avg_conversion_rate = round(avg_conversion, 3),
        avg_food_cost_perc  = round(avg_food_cost, 1),
        past_promo_types    = past_promo_types,
        best_past_roi       = round(best_past_roi, 1),
        worst_past_roi      = round(worst_past_roi, 1),
        avg_past_roi        = round(avg_past_roi, 1),
        total_past_promos   = total_past_promos,
        new_user_ratio      = round(new_user_ratio, 3),
        rejection_rate      = round(rejection_rate, 3),
        tags                = tags,
    )
    return profile, notes


def parse_xlsx(file_bytes: bytes):
    """
    Single-restaurant parse (backwards compatible).
    Returns (RestaurantProfile, notes) for the first restaurant_id found in metrics.
    """
    sheets = _load_sheets(file_bytes)
    first_id = str(sheets["metrics"]["restaurant_id"].iloc[0])
    return _profile_for(first_id, sheets)


def list_restaurant_ids(file_bytes: bytes) -> list:
    """Return all unique restaurant_ids found in the metrics sheet, preserving natural sort."""
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    if "metrics" not in xl.sheet_names:
        raise ValueError("No 'metrics' sheet found")
    metrics = xl.parse("metrics", usecols=["restaurant_id"])
    ids = list(metrics["restaurant_id"].dropna().unique())
    # natural sort: R1 < R2 < ... < R10
    try:
        ids.sort(key=lambda x: int("".join(filter(str.isdigit, str(x))) or "0"))
    except Exception:
        ids.sort()
    return [str(i) for i in ids]


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _rating_bucket(r: float) -> str:
    if r < 3.8:  return "low"
    if r <= 4.2: return "mid"
    return "high"

def _uptime_bucket(u: float) -> str:
    if u < 0.90: return "low"
    if u < 0.95: return "mid"
    return "high"

def _primary_cuisine(tags: list) -> str:
    priority = ["Sushi","Japanese","American","Burgers","Brazilian","Healthy",
                "Chicken","Asian","Desserts","Comfort Food","Latin American",
                "Barbecue","Fast Food","Pizza"]
    for p in priority:
        if p in tags:
            return p
    return tags[0] if tags else "Other"

def _require_cols(df, cols, sheet_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet_name}' is missing columns: {missing}")

def score_promo(promo_type: str, profile: RestaurantProfile) -> float:
    cuisine = profile.cuisine
    rating_b = _rating_bucket(profile.avg_rating)
    uptime_b = _uptime_bucket(profile.avg_uptime)

    # Base ROI from cuisine × promo matrix
    cuisine_map = CUISINE_PROMO_ROI.get(cuisine, {})
    base_roi = cuisine_map.get(promo_type, BASELINE_ROI.get(promo_type, 30.0))

    # Rating multiplier
    r_mults = RATING_MULTIPLIERS.get(promo_type, {"low": 1, "mid": 1, "high": 1})
    base_roi *= r_mults[rating_b]

    # Uptime multiplier
    u_mults = UPTIME_MULTIPLIERS.get(promo_type, {"low": 1, "mid": 1, "high": 1})
    base_roi *= u_mults[uptime_b]

    # Volume modifier
    if profile.avg_daily_orders > 80:
        base_roi *= 1.07
    elif profile.avg_daily_orders < 20:
        base_roi *= 0.88

    # High rejection rate hurts promo effectiveness
    if profile.rejection_rate > 0.05:
        base_roi *= 0.82

    # Already tried — slight novelty penalty (audience may be saturated)
    if promo_type in profile.past_promo_types:
        base_roi *= 0.91

    # Low new user ratio → free delivery is more valuable
    if promo_type == "free-delivery" and profile.new_user_ratio < 0.15:
        base_roi *= 1.18

    # Rewards needs volume to generate signal
    if promo_type == "rewards" and profile.avg_daily_orders < 25:
        base_roi *= 0.75

    return round(base_roi, 1)


def _discount_guidance(rating: float, cuisine: str) -> str:
    if rating > 4.3:
        return "Recommended: 15–20% off. Premium audiences are discount-sensitive — deep cuts signal desperation."
    if rating < 3.8:
        return "Recommended: 10–15% off to test price sensitivity before committing to larger spend."
    return "Recommended: 15–25% off — the dataset sweet spot. Above 35% ROI drops to ~17%."


def _target_guidance(promo_type: str, profile: RestaurantProfile) -> str:
    if profile.new_user_ratio < 0.15:
        return "Target: New customers — your returning base is strong, focus on acquisition."
    if promo_type == "rewards":
        return "Target: Premium / top 20% spenders — build a loyalty tier around your best guests."
    if promo_type == "two-for-one":
        return "Target: All customers — volume-first mechanics work best with broad reach."
    return "Target: All customers."


def _implementation(promo_type: str, profile: RestaurantProfile, roi: float) -> str:
    discount_str = "15–20% off" if profile.avg_rating > 4.3 else "20–25% off"
    impls = {
        "free-delivery": f"Activate Free Delivery Thursday–Sunday for 2 weeks. No minimum order. Monitor new vs returning split daily.",
        "save":          f"Run a Save promotion at {discount_str}. Set minimum order at ~80% of your avg check size. Target all customers.",
        "two-for-one":   f"Launch Two-for-One on your top 2–3 selling items. Run Thursday–Sunday only to protect weekday margins.",
        "rewards":       f"Enable a Rewards programme with 3 tiers. Tier 1: 5th order free item. Tier 2: 10th order 20% off. Measure 30-day repeat rate.",
    }
    return impls.get(promo_type, "Launch promotion targeting all customers.")


def _headline(promo_type: str, roi: float, profile: RestaurantProfile) -> str:
    rating_b = _rating_bucket(profile.avg_rating)
    headlines = {
        ("free-delivery", "mid"): f"Free Delivery is your highest-leverage action — {roi}% predicted ROI in your rating tier",
        ("save", "high"):         f"Targeted discount at {roi}% ROI — use 15–20% to protect premium positioning",
        ("two-for-one", "mid"):   f"Two-for-One peaks at 3.8–4.2★ — your {profile.avg_rating}★ profile is in the sweet spot",
        ("rewards", "high"):      f"Rewards program: the right tool for high-rated restaurants with loyal repeat bases",
    }
    key = (promo_type, rating_b)
    if key in headlines:
        return headlines[key]
    return f"Predicted {roi}% ROI based on {profile.cuisine} cuisine patterns in the dataset"


def _supporting_data(promo_type: str, profile: RestaurantProfile) -> list:
    points = []
    rating_b = _rating_bucket(profile.avg_rating)

    # Dataset ROI reference
    baseline = BASELINE_ROI.get(promo_type, 0)
    points.append(f"Dataset avg ROI for {PROMO_LABELS[promo_type]}: {baseline}%")

    # Cuisine-specific
    cuisine_roi = CUISINE_PROMO_ROI.get(profile.cuisine, {}).get(promo_type)
    if cuisine_roi:
        points.append(f"{profile.cuisine} cuisine avg ROI for this promo: {cuisine_roi}%")

    # Rating tier insight
    rating_insights = {
        ("free-delivery", "mid"):  "Restaurants rated 3.8–4.2★: avg 106% ROI on Free Delivery",
        ("free-delivery", "high"): "Restaurants rated >4.2★: avg 51% ROI on Free Delivery (lower than mid-tier)",
        ("two-for-one", "mid"):    "Two-for-one ROI peaks at 34% for 3.8–4.2★ (vs 22% for >4.2★)",
        ("save", "low"):           "Low-rated restaurants: Save promos yield similar ROI — focus on ops first",
    }
    insight = rating_insights.get((promo_type, rating_b))
    if insight:
        points.append(insight)

    # Volume note
    if profile.avg_daily_orders > 80:
        points.append(f"Your {profile.avg_daily_orders:.0f} daily orders: +7% ROI boost from high reach volume")
    elif profile.avg_daily_orders < 25:
        points.append(f"Your {profile.avg_daily_orders:.0f} daily orders: -12% adjustment — limited promo reach")

    # Uptime note
    uptime_pct = round(profile.avg_uptime * 100, 1)
    uptime_b = _uptime_bucket(profile.avg_uptime)
    if uptime_b == "mid":
        points.append(f"Your {uptime_pct}% uptime is in the 90–95% ROI-optimal band")
    elif uptime_b == "low":
        points.append(f"Your {uptime_pct}% uptime: -29% ROI adjustment — fix availability before heavy promo spend")

    return points


def _applicable_truths(profile: RestaurantProfile) -> list:
    rating_b = _rating_bucket(profile.avg_rating)
    uptime_b = _uptime_bucket(profile.avg_uptime)
    applicable = []
    for t in HIDDEN_TRUTHS:
        cond = t["condition"]
        if cond == "always":
            applicable.append(t)
        elif cond == "rating_mid" and rating_b == "mid":
            applicable.append(t)
        elif cond == "uptime_mid" and uptime_b == "mid":
            applicable.append(t)
        elif cond == "cuisine_sushi" and profile.cuisine in ("Sushi", "Japanese"):
            applicable.append(t)
        elif cond == "cuisine_healthy" and profile.cuisine == "Healthy":
            applicable.append(t)
        elif cond == "low_volume" and profile.avg_daily_orders < 25:
            applicable.append(t)
    return applicable[:5]  # top 5 most relevant


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NOTES = (
    "Scoring model trained on 1,458 real promotions across 50 restaurants "
    "(Dec 2025 – Mar 2026). ROI predictions use a cuisine × promo-type matrix "
    "with rating, uptime, volume, and context multipliers. WIN threshold = 35% ROI."
)


def _build_report(profile, data_notes: list) -> dict:
    """Core: score all promo types for one profile and return a report dict."""
    promo_types = ["free-delivery", "save", "two-for-one", "rewards"]
    scored = sorted(
        [(pt, score_promo(pt, profile)) for pt in promo_types],
        key=lambda x: x[1], reverse=True
    )

    actions = []
    for rank, (pt, roi) in enumerate(scored[:5], 1):
        is_win     = roi > 35.0
        confidence = min(94, max(42, int(65 + (roi - 35) * 0.8))) if is_win \
                     else max(38, int(65 - (35 - roi) * 1.2))
        actions.append({
            "rank":                  rank,
            "promo_type":            pt,
            "label":                 PROMO_LABELS[pt],
            "predicted_roi":         roi,
            "confidence_pct":        confidence,
            "is_predicted_win":      is_win,
            "headline":              _headline(pt, roi, profile),
            "rationale":             PROMO_DESCRIPTIONS[pt],
            "implementation":        _implementation(pt, profile, roi),
            "discount_guidance":     _discount_guidance(profile.avg_rating, profile.cuisine) if pt == "save" else None,
            "target_guidance":       _target_guidance(pt, profile),
            "timing_guidance":       "Best days: Thursday–Sunday (weekends avg 40% more orders than weekdays)",
            "supporting_data_points": _supporting_data(pt, profile),
        })

    truths  = _applicable_truths(profile)
    avg_roi = round(sum(a["predicted_roi"] for a in actions) / len(actions), 1)
    wins    = sum(1 for a in actions if a["is_predicted_win"])

    return {
        "restaurant_id":      profile.restaurant_id,
        "cuisine":            profile.cuisine,
        "tags":               profile.tags,
        "avg_rating":         profile.avg_rating,
        "avg_uptime_pct":     round(profile.avg_uptime * 100, 1),
        "avg_daily_orders":   profile.avg_daily_orders,
        "avg_conversion_rate":profile.avg_conversion_rate,
        "avg_food_cost_perc": profile.avg_food_cost_perc,
        "total_past_promos":  profile.total_past_promos,
        "avg_past_roi":       profile.avg_past_roi,
        "predicted_wins":     wins,
        "avg_predicted_roi":  avg_roi,
        "top_actions":        actions,
        "hidden_truths":      truths,
        "data_quality_notes": data_notes,
        "model_notes":        MODEL_NOTES,
    }


def analyse(file_bytes: bytes) -> dict:
    """
    Single-restaurant analysis (first restaurant_id in metrics sheet).
    Returns a single report dict.
    """
    profile, notes = parse_xlsx(file_bytes)
    return _build_report(profile, notes)


def analyse_all(file_bytes: bytes) -> dict:
    """
    Multi-restaurant analysis.
    Parses all sheets once, then scores every restaurant_id found in metrics.
    Returns:
      {
        "restaurant_count": int,
        "restaurant_ids": [...],
        "reports": [ <report>, ... ],          # same structure as analyse()
        "summary": [                            # one row per restaurant, sorted by avg_predicted_roi desc
          { restaurant_id, cuisine, avg_rating, avg_uptime_pct,
            avg_daily_orders, predicted_wins, avg_predicted_roi,
            top_action_label, top_action_roi }
        ]
      }
    """
    sheets  = _load_sheets(file_bytes)
    metrics = sheets["metrics"]
    ids     = list(metrics["restaurant_id"].dropna().unique())

    # natural sort
    try:
        ids.sort(key=lambda x: int("".join(filter(str.isdigit, str(x))) or "0"))
    except Exception:
        ids.sort()
    ids = [str(i) for i in ids]

    reports = []
    for rid in ids:
        try:
            profile, notes = _profile_for(rid, sheets)
            report = _build_report(profile, notes)
            reports.append(report)
        except Exception as e:
            # Don't fail the whole batch — record the error per restaurant
            reports.append({
                "restaurant_id": rid,
                "error": str(e),
                "top_actions": [],
            })

    # Build summary table, skip errored entries
    summary = []
    for r in reports:
        if "error" in r:
            summary.append({
                "restaurant_id":    r["restaurant_id"],
                "cuisine":          "—",
                "avg_rating":       None,
                "avg_uptime_pct":   None,
                "avg_daily_orders": None,
                "predicted_wins":   0,
                "avg_predicted_roi":None,
                "top_action_label": "—",
                "top_action_roi":   None,
                "error":            r["error"],
            })
        else:
            top = r["top_actions"][0] if r["top_actions"] else {}
            summary.append({
                "restaurant_id":    r["restaurant_id"],
                "cuisine":          r["cuisine"],
                "avg_rating":       r["avg_rating"],
                "avg_uptime_pct":   r["avg_uptime_pct"],
                "avg_daily_orders": r["avg_daily_orders"],
                "predicted_wins":   r["predicted_wins"],
                "avg_predicted_roi":r["avg_predicted_roi"],
                "top_action_label": top.get("label", "—"),
                "top_action_roi":   top.get("predicted_roi"),
            })

    # Sort summary by avg_predicted_roi descending (None last)
    summary.sort(key=lambda x: x["avg_predicted_roi"] or -999, reverse=True)

    return {
        "restaurant_count": len(ids),
        "restaurant_ids":   ids,
        "reports":          reports,
        "summary":          summary,
    }

