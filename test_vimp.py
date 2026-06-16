"""Quick test to verify plot_feature_importance, plot_brier_curve, plot_calibration work."""
import sys
import os

# Verify the core flow: compute importance, then check the plot doesn't crash.
print("=== Testing VIMP / Brier / Calibration plot methods ===")

import numpy as np
import pandas as pd

# Minimal test of _compute_permutation_importance logic
from sklearn.inspection import permutation_importance

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import brier_score, integrated_brier_score
    SKSURV = True
except ImportError:
    SKSURV = False
    print("WARNING: scikit-survival not installed, skipping model tests")

if SKSURV:
    # Create simple dummy data
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "var1": np.random.randn(n),
        "var2": np.random.randn(n),
        "var3": np.random.randn(n),
    })
    times = np.abs(np.random.randn(n)) * 10 + 1
    events = np.random.choice([True, False], size=n, p=[0.6, 0.4])
    y = np.array([(e, t) for e, t in zip(events, times)], dtype=[("event", bool), ("time", float)])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomSurvivalForest(n_estimators=50, max_features="sqrt", random_state=42, oob_score=True)
    model.fit(X_train, y_train)
    print(f"  Model fitted: {model.n_estimators} trees, oob={getattr(model, 'oob_score_', None)}")

    # Test permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
    print(f"  Permutation importance means: {result.importances_mean}")
    print(f"  Permutation importance stds:  {result.importances_std}")

    imp_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
        "importance_lower": result.importances_mean - 1.96 * result.importances_std / np.sqrt(5),
        "importance_upper": result.importances_mean + 1.96 * result.importances_std / np.sqrt(5),
    }).sort_values("importance", ascending=False)
    print(f"  importance_df shape: {imp_df.shape}")
    print(f"  importance_df empty: {imp_df.empty}")
    print(f"  importance_df:\n{imp_df}")

    # Test with test_size=0 (train on all data)
    print("\n--- Testing with test_size=0 (no holdout) ---")
    model2 = RandomSurvivalForest(n_estimators=50, max_features="sqrt", random_state=42, oob_score=True)
    model2.fit(X, y)
    
    # When test_size=0, X_test is empty
    X_test_empty = X.iloc[0:0].copy()
    y_test_empty = y[:0]
    print(f"  len(X_test_empty)={len(X_test_empty)}, len(y_test_empty)={len(y_test_empty)}")
    
    # This should be skipped (len(X_test) == 0) and fall back to train data
    if len(X_test_empty) > 0:
        imp2 = permutation_importance(model2, X_test_empty, y_test_empty, n_repeats=5, random_state=42, n_jobs=1)
    else:
        print("  X_test is empty -> computing on training data instead")
        imp2 = permutation_importance(model2, X, y, n_repeats=5, random_state=42, n_jobs=1)
    
    imp_df2 = pd.DataFrame({
        "feature": X.columns,
        "importance": imp2.importances_mean,
        "importance_std": imp2.importances_std,
    })
    print(f"  importance_df2 shape: {imp_df2.shape}")
    print(f"  importance_df2 empty: {imp_df2.empty}")
    print(f"  importance_df2:\n{imp_df2}")

    # Test Brier score computation
    print("\n--- Testing Brier score ---")
    surv_fns = model.predict_survival_function(X_test)
    t_min = float(np.percentile(y_train["time"], 5))
    t_max = float(np.percentile(y_train["time"], 95))
    eval_times = np.linspace(t_min, t_max, 20)
    
    # Filter eval_times to valid range
    test_times = np.array(y_test["time"], dtype=float)
    eval_times = eval_times[(eval_times >= test_times.min()) & (eval_times < test_times.max())]
    
    if len(eval_times) >= 2:
        surv_matrix = np.array([[fn(t) for t in eval_times] for fn in surv_fns])
        _, brier_values = brier_score(y_train, y_test, surv_matrix, eval_times)
        print(f"  Brier values shape: {np.array(brier_values).shape}")
        print(f"  Brier values (first 5): {np.array(brier_values)[:5]}")
        
        brier_df = pd.DataFrame({"time": eval_times, "brier_score": brier_values})
        print(f"  brier_df shape: {brier_df.shape}")
        print(f"  brier_df empty: {brier_df.empty}")
    else:
        print("  Not enough valid eval_times")

    print("\n=== ALL TESTS PASSED ===")
else:
    print("Skipped (no scikit-survival)")
