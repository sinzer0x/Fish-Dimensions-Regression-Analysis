import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_dataset():
    if os.path.exists("data/processed/fish_checked.csv"):
        df = pd.read_csv("data/processed/fish_checked.csv")
        df = df[df["flag_error"] == 0].copy()
    else:
        df = pd.read_csv("data/Fish.csv")
        df = df[df["Weight"] > 0].copy()
    return df


def get_frozen_split(df):
    features = ["Species", "Length1", "Length2", "Length3", "Height", "Width"]
    X = df[features]
    y = df["Weight"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=X["Species"])


def evaluate_saved_model(model_path="models/best_model.joblib"):
    df = load_dataset()
    _, X_test, _, y_test = get_frozen_split(df)

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=== Итоговые метрики лучшей модели на тесте ===")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f} г")
    print(f"R2 : {r2:.4f}")

    # Анализ ошибок (пункт 6 методички)
    report_df = X_test.copy()
    report_df["y_true"] = y_test.values
    report_df["y_pred"] = np.round(preds, 2)
    report_df["residual"] = np.round(report_df["y_true"] - report_df["y_pred"], 2)
    report_df["abs_error"] = np.round(report_df["residual"].abs(), 2)

    neg_count = int((preds < 0).sum())
    print(f"\nОтрицательных прогнозов: {neg_count}")

    print("\n=== Топ-5 крупнейших абсолютных ошибок ===")
    print(report_df.sort_values("abs_error", ascending=False).head(5).to_string())

    print("\n=== MAE по видам рыб ===")
    species_mae = report_df.groupby("Species")["abs_error"].agg(["count", "mean"]).round(2)
    print(species_mae.to_string())

    print("\n=== MAE по диапазонам массы ===")
    bins = [0, 250, 700, 2000]
    labels = ["Мелкая (<250г)", "Средняя (250-700г)", "Крупная (>700г)"]
    report_df["size_group"] = pd.cut(report_df["y_true"], bins=bins, labels=labels)
    size_mae = report_df.groupby("size_group", observed=False)["abs_error"].agg(["count", "mean"]).round(2)
    print(size_mae.to_string())


if __name__ == "__main__":
    evaluate_saved_model()