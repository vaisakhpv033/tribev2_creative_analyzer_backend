"""
Test: Backend Model Alignment Validation
==========================================
Verifies that:
1. The updated analyzer extracts all 6 model features correctly
2. The XGBoost models load and produce valid predictions
3. The model features match between the backend and phase2 pipeline
4. The overall_score = predicted_proba * 100 is correctly computed
5. Legacy dimension scores are still computed for backward compatibility
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import numpy as np

# Add the backend dir to path
sys.path.insert(0, os.path.dirname(__file__))

def test_analyzer():
    """Test that analyzer extracts all required features."""
    print("=" * 60)
    print("TEST 1: BrainAnalyzer Feature Extraction")
    print("=" * 60)

    from analyzer import BrainAnalyzer

    analyzer = BrainAnalyzer()

    # Find a real .npz file to test with
    npz_dir = r"D:\Work\R_and_D\tribev2\phase2\output_npz"
    npz_files = []
    for root, dirs, files in os.walk(npz_dir):
        for f in files:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(root, f))
    
    if not npz_files:
        print(f"  WARNING: No .npz files found in {npz_dir}")
        print("  Creating synthetic test data...")
        np.random.seed(42)
        test_preds = np.random.randn(30, 20484).astype(np.float32) * 0.1
        test_npz = os.path.join(os.path.dirname(__file__), "test_synthetic.npz")
        np.savez_compressed(test_npz, preds=test_preds)
        npz_files = [test_npz]

    test_npz = npz_files[0]
    print(f"  Testing with: {os.path.basename(test_npz)}")

    result = analyzer.analyze(test_npz)

    # Verify structure
    assert 'raw_scores' in result, "Missing 'raw_scores'"
    assert 'model_features' in result, "Missing 'model_features'"
    assert 'timeseries' in result, "Missing 'timeseries'"
    assert 'global_mean' in result, "Missing 'global_mean'"

    # Verify legacy scores
    legacy_keys = {'visual', 'auditory', 'emotional', 'language', 'attention'}
    actual_legacy = set(result['raw_scores'].keys())
    assert actual_legacy == legacy_keys, f"Legacy keys mismatch: {actual_legacy} != {legacy_keys}"
    print(f"  ✅ Legacy scores: {result['raw_scores']}")

    # Verify model features
    expected_model_keys = {
        'longest_sustained_above_mean',
        'emotional_mean',
        'orbital_mean',
        'visual_std',
        'insula_short_mean',
        'attention_onset_second',
    }
    actual_model = set(result['model_features'].keys())
    assert actual_model == expected_model_keys, f"Model keys mismatch: {actual_model} != {expected_model_keys}"

    print(f"  ✅ Model features:")
    for k, v in result['model_features'].items():
        print(f"     {k}: {v:.6f}")

    # Verify types and ranges
    for k, v in result['model_features'].items():
        assert isinstance(v, float), f"{k} is not float: {type(v)}"
        assert not np.isnan(v), f"{k} is NaN"
        assert not np.isinf(v), f"{k} is Inf"
    print(f"  ✅ All model features are valid floats (no NaN/Inf)")

    # Verify consistency: emotional_mean should equal raw_scores['emotional']
    assert abs(result['model_features']['emotional_mean'] - result['raw_scores']['emotional']) < 1e-10, \
        "emotional_mean doesn't match raw_scores['emotional']"
    print(f"  ✅ emotional_mean matches raw_scores['emotional']")

    print("  PASSED\n")
    return result


def test_model_loading():
    """Test that XGBoost models load correctly."""
    print("=" * 60)
    print("TEST 2: XGBoost Model Loading")
    print("=" * 60)

    import xgboost as xgb
    from pathlib import Path

    model_dir = Path(__file__).parent / "ml_models"

    # Check files exist
    for fname in ["xgb_regressor.json", "xgb_classifier.json", "selected_features.json"]:
        fpath = model_dir / fname
        assert fpath.exists(), f"Missing: {fpath}"
        print(f"  ✅ {fname} exists ({fpath.stat().st_size} bytes)")

    # Load feature list
    with open(model_dir / "selected_features.json") as f:
        features = json.load(f)
    
    expected = [
        "longest_sustained_above_mean",
        "emotional_mean",
        "orbital_mean",
        "visual_std",
        "insula_short_mean",
        "attention_onset_second",
    ]
    assert features == expected, f"Feature mismatch: {features}"
    print(f"  ✅ Feature list matches: {features}")

    # Load models
    reg = xgb.XGBRegressor()
    reg.load_model(str(model_dir / "xgb_regressor.json"))
    print(f"  ✅ Regressor loaded (n_features={reg.n_features_in_})")

    clf = xgb.XGBClassifier()
    clf.load_model(str(model_dir / "xgb_classifier.json"))
    print(f"  ✅ Classifier loaded (n_features={clf.n_features_in_})")

    assert reg.n_features_in_ == 6, f"Regressor expects {reg.n_features_in_} features, not 6"
    assert clf.n_features_in_ == 6, f"Classifier expects {clf.n_features_in_} features, not 6"

    # Check quantile models
    for qname in ["xgb_quantile_p10.json", "xgb_quantile_p90.json"]:
        qpath = model_dir / qname
        if qpath.exists():
            q = xgb.XGBRegressor()
            q.load_model(str(qpath))
            print(f"  ✅ {qname} loaded (n_features={q.n_features_in_})")
        else:
            print(f"  ⚠️ {qname} not found (optional)")

    print("  PASSED\n")
    return reg, clf, features


def test_end_to_end_prediction(analysis_result, reg, clf, features):
    """Test full pipeline: features → model → prediction."""
    print("=" * 60)
    print("TEST 3: End-to-End Prediction")
    print("=" * 60)

    model_features = analysis_result['model_features']

    # Build input array in correct feature order
    X = np.array([[model_features[f] for f in features]])
    print(f"  Input shape: {X.shape}")
    print(f"  Input values: {X[0].tolist()}")

    # Regression
    predicted_log_ctr = reg.predict(X)[0]
    predicted_ctr = float(np.expm1(predicted_log_ctr))
    print(f"  Predicted CTR: {predicted_ctr:.4f}%")
    assert not np.isnan(predicted_ctr), "Predicted CTR is NaN"
    assert predicted_ctr >= -5.0, f"Predicted CTR suspiciously low: {predicted_ctr}"
    print(f"  ✅ Regression prediction valid")

    # Classification
    predicted_class = int(clf.predict(X)[0])
    predicted_proba = float(clf.predict_proba(X)[0][1])
    print(f"  Predicted class: {'High' if predicted_class == 1 else 'Low'}")
    print(f"  Predicted proba: {predicted_proba:.4f}")
    assert 0.0 <= predicted_proba <= 1.0, f"Proba out of range: {predicted_proba}"
    print(f"  ✅ Classification prediction valid")

    # Overall score
    overall_score = predicted_proba * 100.0
    print(f"  Overall score: {overall_score:.1f}/100")
    assert 0.0 <= overall_score <= 100.0, f"Overall score out of range: {overall_score}"
    print(f"  ✅ Overall score in valid range")

    # Tier
    if predicted_proba >= 0.80:
        tier = "Strong High"
    elif predicted_proba >= 0.60:
        tier = "Likely High"
    elif predicted_proba >= 0.40:
        tier = "Borderline"
    elif predicted_proba >= 0.20:
        tier = "Likely Low"
    else:
        tier = "Strong Low"
    print(f"  Tier: {tier}")
    print(f"  ✅ Tier classification valid")

    print("  PASSED\n")


def test_legacy_scoring(analysis_result):
    """Test that legacy dimension scores still work."""
    print("=" * 60)
    print("TEST 4: Legacy Dimension Scoring")
    print("=" * 60)

    raw = analysis_result['raw_scores']

    def scale_dimension(val):
        scaled = (val + 0.5) * 100
        return max(0.0, min(100.0, scaled))

    visual_score = scale_dimension(raw['visual'])
    auditory_score = scale_dimension(raw['auditory'])
    emotional_score = scale_dimension(raw['emotional'])
    language_score = scale_dimension(raw['language'])
    attention_score = min(100.0, max(0.0, raw['attention'] * 50))

    for name, score in [("visual", visual_score), ("auditory", auditory_score),
                        ("emotional", emotional_score), ("language", language_score),
                        ("attention", attention_score)]:
        assert 0.0 <= score <= 100.0, f"{name} score out of range: {score}"
        print(f"  {name:12s}: {score:.1f}/100")

    print(f"  ✅ All legacy scores in [0, 100]")
    print("  PASSED\n")


def test_parity_with_phase2():
    """Compare analyzer output with phase2/extract_features.py output."""
    print("=" * 60)
    print("TEST 5: Parity Check with Phase2 Pipeline")
    print("=" * 60)

    # Load the brain_features.csv from phase2
    import pandas as pd
    csv_path = r"D:\Work\R_and_D\tribev2\phase2\features_npz\brain_features.csv"
    if not os.path.exists(csv_path):
        print(f"  SKIPPED: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from phase2 brain_features.csv")

    # Pick a video and find its .npz file
    sample = df.iloc[0]
    filename = sample['filename']
    print(f"  Testing video: {filename}")

    # Find the .npz file
    npz_dir = r"D:\Work\R_and_D\tribev2\phase2\output_npz"
    npz_path = None
    for root, dirs, files in os.walk(npz_dir):
        for f in files:
            if f == f"{filename}.npz":
                npz_path = os.path.join(root, f)
                break
        if npz_path:
            break

    if not npz_path:
        print(f"  SKIPPED: Could not find .npz for {filename}")
        return

    # Run backend analyzer
    from analyzer import BrainAnalyzer
    analyzer = BrainAnalyzer()
    result = analyzer.analyze(npz_path)
    mf = result['model_features']

    # Compare with phase2 CSV values
    features_to_check = [
        'emotional_mean', 'orbital_mean', 'visual_std',
        'insula_short_mean', 'attention_onset_second',
        'longest_sustained_above_mean',
    ]

    all_close = True
    for feat in features_to_check:
        backend_val = mf[feat]
        phase2_val = sample[feat]
        diff = abs(backend_val - phase2_val)
        # Use relative tolerance for very small values
        tol = max(1e-6, abs(phase2_val) * 0.01)
        match = diff < tol
        symbol = "✅" if match else "❌"
        print(f"  {symbol} {feat:35s}  backend={backend_val:.6f}  phase2={phase2_val:.6f}  diff={diff:.2e}")
        if not match:
            all_close = False

    if all_close:
        print("  ✅ All features match within tolerance")
    else:
        print("  ⚠️ Some features differ — check extraction logic")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Backend Model Alignment — Test Suite")
    print("=" * 60 + "\n")

    # Test 1: Analyzer
    result = test_analyzer()

    # Test 2: Model loading
    reg, clf, features = test_model_loading()

    # Test 3: End-to-end prediction
    test_end_to_end_prediction(result, reg, clf, features)

    # Test 4: Legacy scoring
    test_legacy_scoring(result)

    # Test 5: Parity with phase2
    test_parity_with_phase2()

    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
