"""
End-to-end test for the Gemini insights pipeline.
Tests: prompts → LLM client → structured output → validation
"""
import sys
import os
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load env
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))


def test_prompt_formatting():
    """Test that prompt formatting produces valid output."""
    print("=" * 60)
    print("TEST 1: Prompt Formatting")
    print("=" * 60)

    from prompts import SYSTEM_PROMPT, format_analysis_data, FEATURE_DOCS

    assert len(FEATURE_DOCS) == 6, f"Expected 6 features, got {len(FEATURE_DOCS)}"
    print(f"  ✅ 6 feature docs defined")

    assert len(SYSTEM_PROMPT) > 1000, f"System prompt too short: {len(SYSTEM_PROMPT)}"
    print(f"  ✅ System prompt: {len(SYSTEM_PROMPT)} chars")

    user_msg = format_analysis_data(
        video_name="test_video.mp4",
        predicted_ctr=4.82,
        predicted_class="High",
        predicted_proba=0.725,
        prediction_tier="Likely High",
        overall_score=72.5,
        feature_values={
            "longest_sustained_above_mean": 22.0,
            "emotional_mean": 0.025,
            "orbital_mean": 0.018,
            "visual_std": 0.08,
            "insula_short_mean": 0.012,
            "attention_onset_second": 5.0,
        },
        ctr_lower_bound=2.1,
        ctr_upper_bound=7.3,
        dimension_scores={
            "visual": 65.0,
            "auditory": 55.0,
            "emotional": 52.5,
            "attention": 70.0,
            "language": 48.0,
        },
    )

    assert "test_video.mp4" in user_msg
    assert "4.82%" in user_msg
    assert "orbital_mean" in user_msg
    print(f"  ✅ User message: {len(user_msg)} chars")
    print(f"  Preview:\n{user_msg[:300]}...")
    print("  PASSED\n")
    return user_msg


def test_gemini_call(user_msg: str):
    """Test actual Gemini API call with structured output."""
    print("=" * 60)
    print("TEST 2: Gemini API Call (LIVE)")
    print("=" * 60)

    from prompts import SYSTEM_PROMPT
    from llm_client import GeminiClient

    client = GeminiClient()
    print(f"  Model: {client._model}")
    print(f"  Making API call...")

    result = client.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_msg,
        temperature=0.7,
    )

    print(f"  ✅ Got response with keys: {list(result.keys())}")

    # Validate structure
    assert "summary" in result, "Missing 'summary'"
    assert isinstance(result["summary"], str), "'summary' not a string"
    assert len(result["summary"]) > 20, f"Summary too short: {result['summary']}"
    print(f"  ✅ Summary: {result['summary'][:100]}...")

    assert "strengths" in result, "Missing 'strengths'"
    assert isinstance(result["strengths"], list), "'strengths' not a list"
    assert len(result["strengths"]) >= 1, "No strengths found"
    print(f"  ✅ Strengths: {len(result['strengths'])} items")
    for s in result["strengths"]:
        assert "title" in s, f"Strength missing 'title': {s}"
        assert "description" in s, f"Strength missing 'description': {s}"
        assert "impact" in s, f"Strength missing 'impact': {s}"

    assert "weaknesses" in result, "Missing 'weaknesses'"
    assert isinstance(result["weaknesses"], list), "'weaknesses' not a list"
    print(f"  ✅ Weaknesses: {len(result['weaknesses'])} items")

    assert "recommendations" in result, "Missing 'recommendations'"
    assert isinstance(result["recommendations"], list), "'recommendations' not a list"
    assert len(result["recommendations"]) >= 1, "No recommendations found"
    print(f"  ✅ Recommendations: {len(result['recommendations'])} items")
    for r in result["recommendations"]:
        assert "title" in r, f"Recommendation missing 'title': {r}"
        assert "expected_impact" in r, f"Recommendation missing 'expected_impact': {r}"

    assert "feature_analysis" in result, "Missing 'feature_analysis'"
    assert isinstance(result["feature_analysis"], list), "'feature_analysis' not a list"
    print(f"  ✅ Feature analyses: {len(result['feature_analysis'])} items")
    for f in result["feature_analysis"]:
        assert "feature_name" in f, f"Feature analysis missing 'feature_name': {f}"
        assert "rating" in f, f"Feature analysis missing 'rating': {f}"
        assert f["rating"] in ("excellent", "good", "average", "poor"), \
            f"Invalid rating '{f['rating']}' for {f['feature_name']}"

    print("  PASSED\n")

    # Pretty print the full result
    print("=" * 60)
    print("FULL GEMINI RESPONSE:")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print()

    return result


def test_validation(result: dict):
    """Test the insight validation function."""
    print("=" * 60)
    print("TEST 3: Response Validation")
    print("=" * 60)

    from insights_service import _validate_insight_response

    _validate_insight_response(result)
    print("  ✅ Validation passed")

    # Test missing keys
    try:
        _validate_insight_response({"summary": "test"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✅ Missing keys correctly caught: {e}")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Gemini Insights Pipeline — End-to-End Test")
    print("=" * 60 + "\n")

    user_msg = test_prompt_formatting()
    result = test_gemini_call(user_msg)
    test_validation(result)

    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
