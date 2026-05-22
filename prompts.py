"""
Prompt Templates for Creative Insights
========================================
Contains the system prompt, data formatter, and response schema
for generating structured creative analysis via Gemini LLM.

Separation of concerns:
  - This module ONLY handles prompt construction and data formatting.
  - LLM communication is handled by llm_client.py
  - Orchestration is handled by insights_service.py
"""

from __future__ import annotations
from typing import Any


# ── Feature Documentation ────────────────────────────────────────────────
# Maps each model feature to its human-readable description,
# neuroscience explanation, and correlation with CTR.

FEATURE_DOCS = {
    "longest_sustained_above_mean": {
        "label": "Sustained Engagement",
        "brain_region": "Whole-brain (global mean)",
        "unit": "seconds",
        "correlation": "+0.587",
        "direction": "positive",
        "explanation": (
            "The longest consecutive streak of seconds where whole-brain "
            "activation stays above the video's mean activation level. "
            "Higher values indicate the ad holds sustained neural engagement "
            "without dipping. This is one of the strongest CTR predictors."
        ),
    },
    "emotional_mean": {
        "label": "Emotional Response",
        "brain_region": "Orbitofrontal cortex, insula, anterior cingulate (grouped emotional regions)",
        "unit": "activation (z-score)",
        "correlation": "+0.584",
        "direction": "positive",
        "explanation": (
            "Average activation of emotional brain regions over the full video. "
            "Higher values mean the ad triggers stronger emotional processing — "
            "both positive (desire, excitement) and negative (urgency, fear). "
            "Emotional ads are remembered and acted upon more."
        ),
    },
    "orbital_mean": {
        "label": "Reward Center Activation",
        "brain_region": "G_orbital (orbitofrontal cortex)",
        "unit": "activation (z-score)",
        "correlation": "+0.588",
        "direction": "positive",
        "explanation": (
            "Mean activation of the orbitofrontal cortex — the brain's "
            "reward valuation center. This region fires when the brain "
            "evaluates 'I want that' or 'this is valuable'. The strongest "
            "single predictor of CTR."
        ),
    },
    "visual_std": {
        "label": "Visual Rhythm / Consistency",
        "brain_region": "Visual cortex (occipital lobe regions)",
        "unit": "std dev of activation",
        "correlation": "−0.508",
        "direction": "negative",
        "explanation": (
            "Standard deviation of visual cortex activation over time. "
            "LOWER variability (more consistent visual stimulation) correlates "
            "with HIGHER CTR. Wild visual fluctuations may distract from the "
            "message. A steady visual narrative keeps the brain focused."
        ),
    },
    "insula_short_mean": {
        "label": "Gut Feeling / Visceral Response",
        "brain_region": "G_insular_short (short insular gyrus)",
        "unit": "activation (z-score)",
        "correlation": "+0.490",
        "direction": "positive",
        "explanation": (
            "Mean activation of the short insular gyrus, which processes "
            "gut-level, visceral emotional responses. This is the 'instinct' "
            "region — it fires when content creates an immediate, bodily "
            "reaction (excitement, discomfort, desire). Higher activation "
            "means the ad 'hits you in the gut'."
        ),
    },
    "attention_onset_second": {
        "label": "Attention Build-up Speed",
        "brain_region": "Whole-brain (global mean > threshold)",
        "unit": "seconds",
        "correlation": "+0.520",
        "direction": "positive",
        "explanation": (
            "The first second where brain activation exceeds the engagement "
            "threshold (mean + 0.5×std). A LATER onset can indicate a "
            "suspense/build-up narrative structure that keeps viewers "
            "watching to discover the payoff. However, very late onset "
            "may mean the ad fails to engage at all."
        ),
    },
}


# ── System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert creative strategist and neuroscience analyst for a digital advertising agency. \
You specialize in analyzing brain-based engagement signals from ad videos to provide actionable \
creative recommendations.

## Context

You are analyzing brain activation data from TRIBEv2, a brain simulation model that predicts \
how a human brain would respond to watching a video advertisement. The model uses Meta's \
Trimodal Brain Encoder (V-JEPA2, Wav2Vec-BERT, LLaMA 3.2) to predict activation across 20,484 \
brain surface vertices at 1-second resolution.

An XGBoost model was trained on real ad campaign CTR (Click-Through Rate) data to identify which \
brain signals predict actual ad performance. It selected 6 key brain features based on Spearman \
correlation analysis.

## The 6 CTR-Predictive Brain Features

1. **Reward Center (orbital_mean)** — ρ = +0.588
   Orbitofrontal cortex activation. The brain's "I want that" signal. Higher = better.

2. **Sustained Engagement (longest_sustained_above_mean)** — ρ = +0.587
   Longest streak of above-average brain activation. More seconds = better.

3. **Emotional Response (emotional_mean)** — ρ = +0.584
   Grouped emotional region activation. Higher = stronger emotional processing.

4. **Attention Build-up (attention_onset_second)** — ρ = +0.520
   When strong engagement first kicks in. Later onset can indicate effective suspense narrative.

5. **Visual Rhythm (visual_std)** — ρ = −0.508
   Visual cortex variability. LOWER = better (consistent visuals beat chaotic edits).

6. **Gut Feeling (insula_short_mean)** — ρ = +0.490
   Visceral emotional response. Higher = content creates instinctive reactions.

## Your Task

Given a video's brain feature values and CTR prediction, provide a structured creative analysis. \
Be specific, actionable, and grounded in the neuroscience data. Don't be generic — reference the \
actual values and what they mean for this specific ad.

## Response Format

You MUST respond with valid JSON matching this exact structure:

```json
{
  "summary": "A 2-3 sentence executive summary of the ad's neural performance and predicted effectiveness.",
  "strengths": [
    {
      "title": "Short strength title",
      "description": "Detailed explanation of why this is a strength, referencing specific brain data.",
      "impact": "high|medium|low"
    }
  ],
  "weaknesses": [
    {
      "title": "Short weakness title",
      "description": "Detailed explanation of the weakness and its impact on CTR.",
      "impact": "high|medium|low"
    }
  ],
  "recommendations": [
    {
      "title": "Short recommendation title",
      "description": "Specific, actionable recommendation to improve the ad's neural engagement.",
      "priority": "high|medium|low",
      "expected_impact": "Brief description of expected improvement"
    }
  ],
  "feature_analysis": [
    {
      "feature_name": "orbital_mean",
      "value": 0.025,
      "rating": "excellent|good|average|poor",
      "interpretation": "What this specific value means for this ad's performance."
    }
  ]
}
```

Guidelines:
- Provide 2-4 strengths, 1-3 weaknesses, and 2-4 recommendations
- Each feature_analysis entry should cover one of the 6 features
- Use the actual data values in your analysis
- Be specific and actionable, not generic
- Reference neuroscience concepts when explaining features
- Consider the interplay between features (e.g., strong reward + strong emotion = synergy)
"""


# ── Data Formatter ────────────────────────────────────────────────────────

def format_analysis_data(
    video_name: str,
    predicted_ctr: float,
    predicted_class: str,
    predicted_proba: float,
    prediction_tier: str,
    overall_score: float,
    feature_values: dict[str, float],
    ctr_lower_bound: float | None = None,
    ctr_upper_bound: float | None = None,
    dimension_scores: dict[str, float] | None = None,
) -> str:
    """Format neural analysis data into a structured prompt for the LLM.

    Args:
        video_name: Original filename of the video.
        predicted_ctr: XGBoost predicted CTR percentage.
        predicted_class: "High" or "Low".
        predicted_proba: Probability of being High CTR (0.0–1.0).
        prediction_tier: "Strong High", "Likely High", etc.
        overall_score: Neural score (predicted_proba × 100).
        feature_values: Dict of the 6 model feature values.
        ctr_lower_bound: Optional P10 quantile CTR bound.
        ctr_upper_bound: Optional P90 quantile CTR bound.
        dimension_scores: Optional legacy dimension scores (0-100).

    Returns:
        Formatted string ready to be used as user message to the LLM.
    """
    lines = []
    lines.append(f"## Video: {video_name}")
    lines.append("")

    # ── Prediction Summary ──
    lines.append("### CTR Prediction Results")
    lines.append(f"- **Predicted CTR**: {predicted_ctr:.2f}%")
    if ctr_lower_bound is not None and ctr_upper_bound is not None:
        lines.append(f"- **CTR Range (P10-P90)**: {ctr_lower_bound:.2f}% — {ctr_upper_bound:.2f}%")
    lines.append(f"- **Classification**: {predicted_class} CTR")
    lines.append(f"- **Confidence**: {predicted_proba:.1%} probability of above-average CTR")
    lines.append(f"- **Tier**: {prediction_tier}")
    lines.append(f"- **Neural Score**: {overall_score:.0f}/100")
    lines.append("")

    # ── Feature Values ──
    lines.append("### Brain Feature Values")
    lines.append("")

    feature_order = [
        "orbital_mean", "longest_sustained_above_mean", "emotional_mean",
        "attention_onset_second", "visual_std", "insula_short_mean",
    ]

    for feat_key in feature_order:
        if feat_key not in feature_values:
            continue
        val = feature_values[feat_key]
        doc = FEATURE_DOCS.get(feat_key, {})
        label = doc.get("label", feat_key)
        corr = doc.get("correlation", "?")
        unit = doc.get("unit", "")

        if feat_key in ("longest_sustained_above_mean", "attention_onset_second"):
            val_str = f"{val:.0f} {unit}"
        else:
            val_str = f"{val:.6f} {unit}"

        lines.append(f"- **{label}** ({feat_key}): {val_str}  [ρ = {corr} with CTR]")

    lines.append("")

    # ── Legacy Dimension Scores (optional context) ──
    if dimension_scores:
        lines.append("### Dimension Scores (0-100 scale, for context)")
        for dim, score in dimension_scores.items():
            lines.append(f"- {dim.capitalize()}: {score:.0f}/100")
        lines.append("")

    lines.append("Please analyze this video's neural data and provide your structured assessment.")

    return "\n".join(lines)
