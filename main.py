import numpy as np

from src.data_loader import load_panel, chronological_split, FEATURE_COLS
from src.models import (
    make_logistic_model,
    make_tree_model,
    naive_carry_forward_baseline,
    majority_class_baseline,
)
from src.evaluation import (
    evaluate_predictions,
    plot_confusion,
    find_best_threshold,
)
from src.interpretation import plot_tree_importance

from sklearn.metrics import accuracy_score


def main():
    print("=== Phase 2: Modeling pipeline started ===")
    print("=== Phase 3: Evaluation & Baselines started ===")

    # Load data
    df = load_panel("data/processed/oecd_panel_2010_2023.csv")

    print("\n[DEBUG] Data quality")
    print("Rows:", len(df), "Cols:", df.shape[1])
    print(df[FEATURE_COLS + ["decoupled"]].isnull().sum())
    print(df.dtypes)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        df_train,
        df_val,
        df_test,
    ) = chronological_split(df)
    
    # DEBUG — split integrity 
    print("\n[DEBUG] Years by split")
    print("train years:", sorted(df_train["year"].unique())[:3], "...",
      sorted(df_train["year"].unique())[-3:])
    print("val years  :", sorted(df_val["year"].unique()))
    print("test years :", sorted(df_test["year"].unique()))

    print("\n[DEBUG] Key overlap check")
    train_keys = set(zip(df_train["iso3"], df_train["year"]))
    test_keys  = set(zip(df_test["iso3"], df_test["year"]))
    print("overlap train/test keys:", len(train_keys & test_keys))

    print("\n[DEBUG] Split shapes")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)
    print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    print("\n[DEBUG] Class balance")
    print("train mean decoupled:", float(np.mean(y_train)))
    print("val   mean decoupled:", float(np.mean(y_val)))
    print("test  mean decoupled:", float(np.mean(y_test)))

    # Train models
    print("\nTraining models...")
    log_model = make_logistic_model()
    tree_model = make_tree_model()

    log_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)

    # Overfitting check (tree)
    print("\n[DEBUG] Train vs test accuracy (overfitting check)")
    tree_train_acc = tree_model.score(X_train, y_train)
    tree_test_acc = tree_model.score(X_test, y_test)
    print(
        f"Tree - train: {tree_train_acc:.3f}, test: {tree_test_acc:.3f}, "
        f"gap: {tree_train_acc - tree_test_acc:.3f}"
    )

    print("\nEvaluating on test set (2023)...")

    # Logistic: threshold selected on VAL (2022)
    val_proba = log_model.predict_proba(X_val)[:, 1]
    best_t, best_val_f1 = find_best_threshold(y_val.to_numpy(), val_proba)

    print("\n[DEBUG] Logistic train vs test using tuned threshold")
    train_proba = log_model.predict_proba(X_train)[:, 1]
    test_proba = log_model.predict_proba(X_test)[:, 1]
    y_pred_train = (train_proba >= best_t).astype(int)
    y_pred_test = (test_proba >= best_t).astype(int)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Logistic - train: {train_acc:.3f}, test: {test_acc:.3f}, gap: {train_acc - test_acc:.3f}")
    print(f"logistic: best threshold on val={best_t:.2f} (val f1={best_val_f1:.3f})")

    # Final test predictions
    y_pred_log = y_pred_test
    y_pred_tree = tree_model.predict(X_test)

    # Baselines
    maj = majority_class_baseline(y_train)

    y_pred_base = naive_carry_forward_baseline(
        df_test=df_test,
        df_full=df,
        target_col="decoupled",
        majority_class=maj,
    )

    y_pred_majority = np.full(shape=len(y_test), fill_value=maj, dtype=int)

    # Evaluation
    evaluate_predictions(y_test, y_pred_log, "logistic")
    evaluate_predictions(y_test, y_pred_tree, "tree")
    evaluate_predictions(y_test, y_pred_base, "baseline_carry_forward")
    evaluate_predictions(y_test, y_pred_majority, "baseline_majority_class")

    plot_confusion(y_test, y_pred_majority, "baseline_majority_class")
    plot_confusion(y_test, y_pred_log, "logistic")
    plot_confusion(y_test, y_pred_tree, "tree")
    plot_confusion(y_test, y_pred_base, "baseline_carry_forward")

    plot_tree_importance(tree_model, FEATURE_COLS)
    print("Saved: results/figures/tree_importances.png")

    print("\n=== Phase 2 completed successfully ===")
    print("=== Phase 3 completed successfully ===")


if __name__ == "__main__":
    main()
