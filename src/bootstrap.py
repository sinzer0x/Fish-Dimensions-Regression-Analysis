import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src.evaluate import get_frozen_split, load_dataset
from src.features import VolumeAdder
from src.train import build_preprocessor


def bootstrap_intervals(estimator, X_train, y_train, X_test, n_boot=500, alpha=0.05, seed=42):
    """Оценка 95% доверительного (CI) и предиктивного (PI) интервалов методом бутстрэпа."""
    rng = np.random.default_rng(seed)
    n_train = len(X_train)
    n_test = len(X_test)

    mean_preds = np.zeros((n_boot, n_test))
    pred_samples = np.zeros((n_boot, n_test))

    for b in range(n_boot):
        idx = rng.integers(0, n_train, size=n_train)
        X_b = X_train.iloc[idx]
        y_b = y_train.iloc[idx]

        model_b = clone(estimator)
        model_b.fit(X_b, y_b)

        # Прогноз на тесте
        y_hat_test = model_b.predict(X_test)
        mean_preds[b] = y_hat_test

        # Бутстрэп остатков для предиктивного интервала
        train_residuals = y_b.values - model_b.predict(X_b)
        sampled_res = rng.choice(train_residuals, size=n_test, replace=True)
        pred_samples[b] = y_hat_test + sampled_res

    low_q = 100 * (alpha / 2)
    high_q = 100 * (1 - alpha / 2)

    ci_low, ci_high = np.percentile(mean_preds, [low_q, high_q], axis=0)
    pi_low, pi_high = np.percentile(pred_samples, [low_q, high_q], axis=0)
    point_pred = np.median(mean_preds, axis=0)

    return pd.DataFrame({
        "y_pred": point_pred,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "pi_low": pi_low,
        "pi_high": pi_high,
        "pi_width": pi_high - pi_low,
    })


if __name__ == "__main__":
    df = load_dataset()
    X_train, X_test, y_train, y_test = get_frozen_split(df)

    num_cols = ["Length1", "Length2", "Length3", "Height", "Width"]
    cat_cols = ["Species"]

    pipe_e1 = Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("regressor", LinearRegression()),
    ])
    pipe_e4 = Pipeline([
        ("add_volume", VolumeAdder()),
        ("preprocess", build_preprocessor(num_cols + ["V_proxy"], cat_cols)),
        ("regressor", Ridge(alpha=1.0)),
    ])

    res_e1 = bootstrap_intervals(pipe_e1, X_train, y_train, X_test)
    res_e4 = bootstrap_intervals(pipe_e4, X_train, y_train, X_test)

    for name, res in [("E1_LinearRegression", res_e1), ("E4_Ridge_With_Volume", res_e4)]:
        coverage = np.mean((y_test.values >= res["pi_low"]) & (y_test.values <= res["pi_high"])) * 100
        print(f"=== {name} ===")
        print(f"Средняя ширина 95% ДИ (среднего прогноза) : {res['ci_width'].mean():.2f} г")
        print(f"Средняя ширина 95% ПИ (индивид. прогноза): {res['pi_width'].mean():.2f} г")
        print(f"Фактическое покрытие 95% ПИ на тесте     : {coverage:.1f}%\n")

    print("=== Примеры 95% интервалов лучшей модели (E4) на первых 5 рыбах теста ===")
    preview = X_test[["Species", "Length3"]].head(5).copy().reset_index(drop=True)
    preview["y_true"] = y_test.values[:5]
    preview["y_pred"] = res_e4["y_pred"].head(5).round(1)
    preview["95%_CI"] = [f"[{l:.1f}; {h:.1f}]" for l, h in zip(res_e4["ci_low"].head(5), res_e4["ci_high"].head(5))]
    preview["95%_PI"] = [f"[{l:.1f}; {h:.1f}]" for l, h in zip(res_e4["pi_low"].head(5), res_e4["pi_high"].head(5))]
    print(preview.to_string(index=False))