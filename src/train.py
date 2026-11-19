import os
import platform
import subprocess
import tempfile
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from src.features import VolumeAdder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler




def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def log_and_evaluate(run_name, estimator, X_train, y_train, X_test, y_test, cv, params_dict, feature_list):
    with mlflow.start_run(run_name=run_name):
        # Теги окружения и коммита
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("python_version", platform.python_version())

        # Параметры эксперимента
        mlflow.log_params(params_dict)
        mlflow.log_param("features", ",".join(feature_list))
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Перекрёстная проверка внутри обучающей выборки
        scoring = {
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
        cv_res = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring)

        # Обучение на всей train-выборке
        estimator.fit(X_train, y_train)
        best_model = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator

        if hasattr(estimator, "best_params_"):
            mlflow.log_params({f"best_{k}": v for k, v in estimator.best_params_.items()})

        # Прогноз и метрики на замороженном тесте
        preds = best_model.predict(X_test)
        test_mse = float(mean_squared_error(y_test, preds))
        test_mae = float(mean_absolute_error(y_test, preds))
        test_r2 = float(r2_score(y_test, preds))

        mlflow.log_metrics({
            "cv_mse_mean": float(-cv_res["test_mse"].mean()),
            "cv_mse_std": float(cv_res["test_mse"].std()),
            "cv_mae_mean": float(-cv_res["test_mae"].mean()),
            "cv_mae_std": float(cv_res["test_mae"].std()),
            "cv_r2_mean": float(cv_res["test_r2"].mean()),
            "cv_r2_std": float(cv_res["test_r2"].std()),
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        })

        # Сохранение таблицы прогнозов и остатков как артефакта
        res_df = X_test.copy()
        res_df["y_true"] = y_test.values
        res_df["y_pred"] = preds
        res_df["residual"] = res_df["y_true"] - res_df["y_pred"]
        res_df["abs_error"] = res_df["residual"].abs()

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_path = os.path.join(tmpdir, f"{run_name}_predictions.csv")
            res_df.to_csv(pred_path, index=False)
            mlflow.log_artifact(pred_path, artifact_path="predictions")

        # Логирование пайплайна модели
        mlflow.sklearn.log_model(best_model, name="model", serialization_format="cloudpickle")

        print(f"{run_name:28s} | CV R2: {cv_res['test_r2'].mean():.4f} | Test MSE: {test_mse:8.2f} | Test MAE: {test_mae:6.2f} | Test R2: {test_r2:.4f}")
        return best_model, test_r2


if __name__ == "__main__":
    # Локальный журнал MLflow в папке mlruns
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("FishGrow_Regression")

    # 1. Загрузка очищенных данных
    if os.path.exists("data/processed/fish_checked.csv"):
        df = pd.read_csv("data/processed/fish_checked.csv")
        df = df[df["flag_error"] == 0].copy()
        data_ver = "fish_checked.csv (no Weight=0)"
    else:
        df = pd.read_csv("data/Fish.csv")
        df = df[df["Weight"] > 0].copy()
        data_ver = "Fish.csv (filtered Weight>0)"

    base_features = ["Species", "Length1", "Length2", "Length3", "Height", "Width"]
    X = df[base_features]
    y = df["Weight"]

    # 2. Замороженное разбиение (стратификация по Species)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["Species"]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    num_cols = ["Length1", "Length2", "Length3", "Height", "Width"]
    cat_cols = ["Species"]

    print("Запуск экспериментов E0 - E4...\n")

    # E0: Наивный прогноз среднего
    pipe_e0 = Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("regressor", DummyRegressor(strategy="mean")),
    ])
    log_and_evaluate("E0_Dummy_Mean", pipe_e0, X_train, y_train, X_test, y_test, cv,
                     {"model": "DummyRegressor", "strategy": "mean", "data": data_ver}, base_features)

    # E1: Базовая линейная регрессия
    pipe_e1 = Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("regressor", LinearRegression()),
    ])
    log_and_evaluate("E1_LinearRegression", pipe_e1, X_train, y_train, X_test, y_test, cv,
                     {"model": "LinearRegression", "target_transform": "none", "data": data_ver}, base_features)

    # E1_log: Линейная регрессия с log(1+y) и обратным преобразованием expm1
    pipe_e1_log = Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("regressor", TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1
        )),
    ])
    log_and_evaluate("E1_Linear_LogTarget", pipe_e1_log, X_train, y_train, X_test, y_test, cv,
                     {"model": "LinearRegression", "target_transform": "log1p", "data": data_ver}, base_features)

    # E2: Гребневая регрессия (Ridge) с подбором alpha по CV
    pipe_e2 = Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("regressor", Ridge()),
    ])
    grid_e2 = GridSearchCV(
        pipe_e2,
        param_grid={"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    model_e2, r2_e2 = log_and_evaluate("E2_Ridge_CV", grid_e2, X_train, y_train, X_test, y_test, cv,
                                       {"model": "Ridge", "alpha_grid": "[0.01..100]", "data": data_ver}, base_features)

    # E3: Нелинейная модель (RandomForest) с ограниченным бюджетом поиска
    pipe_e3 = Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("regressor", RandomForestRegressor(random_state=42)),
    ])
    grid_e3 = GridSearchCV(
        pipe_e3,
        param_grid={
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 6, 10],
            "regressor__min_samples_leaf": [1, 2],
        },
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    model_e3, r2_e3 = log_and_evaluate("E3_RandomForest_CV", grid_e3, X_train, y_train, X_test, y_test, cv,
                                       {"model": "RandomForestRegressor", "budget": "12_fits", "data": data_ver}, base_features)

    # E4: Ridge + производный геометрический признак V_proxy = Length3 * Height * Width
    num_cols_vol = num_cols + ["V_proxy"]
    pipe_e4 = Pipeline([
        ("add_volume", VolumeAdder()),
        ("preprocess", build_preprocessor(num_cols_vol, cat_cols)),
        ("regressor", Ridge()),
    ])
    grid_e4 = GridSearchCV(
        pipe_e4,
        param_grid={"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    model_e4, r2_e4 = log_and_evaluate("E4_Ridge_With_Volume", grid_e4, X_train, y_train, X_test, y_test, cv,
                                       {"model": "Ridge+V_proxy", "alpha_grid": "[0.01..100]", "data": data_ver}, base_features + ["V_proxy"])

    # Сохраняем лучшую модель для evaluate.py и тестов
    os.makedirs("models", exist_ok=True)
    candidates = [("E2_Ridge", model_e2, r2_e2), ("E3_RF", model_e3, r2_e3), ("E4_Ridge_Vol", model_e4, r2_e4)]
    best_name, best_pipe, best_score = max(candidates, key=lambda item: item[2])
    joblib.dump(best_pipe, "models/best_model.joblib")
    print(f"\nЛучшая модель ({best_name}, Test R2 = {best_score:.4f}) сохранена в models/best_model.joblib")