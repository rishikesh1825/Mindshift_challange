# ==========================================
# IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import os, glob, warnings
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna

warnings.filterwarnings("ignore")


# ==========================================
# SETTINGS
# ==========================================
SETTINGS = {
    "INPUT_DIR": "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge",
    "FEATURE_PATH": "/kaggle/input/datasets/racharladhanush/catched-features/cached_features.parquet",
    "ID_MAP": "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge/id_mapping_new.csv",

    "CAT_COLS": ['vehicle', 'shift', 'operator_id'],
    "NUM_COLS": [
        'mean_speed', 'max_speed', 'mean_altitude', 'std_altitude',
        'max_altitude', 'min_altitude', 'idle_pings', 'total_pings',
        'min_cumdist', 'max_cumdist', 'distance_travelled', 'net_lift'
    ]
}


# ==========================================
# DATA FUNCTIONS
# ==========================================
def read_inputs():
    print("📥 Reading data...")

    id_map = pd.read_csv(SETTINGS["ID_MAP"])

    files = glob.glob(os.path.join(SETTINGS["INPUT_DIR"], "smry_*_train_ordered.csv"))
    if not files:
        files = glob.glob(os.path.join(SETTINGS["INPUT_DIR"], "*", "smry_*_train_ordered.csv"))

    train = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    features = pd.read_parquet(SETTINGS["FEATURE_PATH"])

    return id_map, train, features


def format_dates(*dfs):
    for df in dfs:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')


def join_data(train, features, id_map):
    train_df = train.merge(features, on=['vehicle', 'date', 'shift'], how='left')
    test_df = id_map.merge(features, on=['vehicle', 'date', 'shift'], how='left')
    return train_df, test_df


# ==========================================
# FEATURE PREP
# ==========================================
def prepare_features(train_df, test_df):

    cat_cols = SETTINGS["CAT_COLS"]
    num_cols = SETTINGS["NUM_COLS"]
    all_cols = cat_cols + num_cols

    # Handle categorical
    for c in cat_cols:
        train_df[c] = train_df[c].fillna("UNKNOWN").astype(str)
        test_df[c] = test_df[c].fillna("UNKNOWN").astype(str)

    train_df = train_df.dropna(subset=['acons']).reset_index(drop=True)

    X = train_df[all_cols]
    y = train_df['acons']
    X_test = test_df[all_cols]

    return X, y, X_test


def build_model_inputs(X, X_test):

    # CatBoost
    X_cb, X_test_cb = X.copy(), X_test.copy()

    # LightGBM
    X_lgbm, X_test_lgbm = X.copy(), X_test.copy()
    for c in SETTINGS["CAT_COLS"]:
        X_lgbm[c] = X_lgbm[c].astype('category')
        X_test_lgbm[c] = pd.Categorical(X_test_lgbm[c], categories=X_lgbm[c].cat.categories)

    # XGBoost
    X_xgb, X_test_xgb = X_lgbm.copy(), X_test_lgbm.copy()
    for c in SETTINGS["CAT_COLS"]:
        X_xgb[c] = X_xgb[c].cat.codes
        X_test_xgb[c] = X_test_xgb[c].cat.codes

    return (X_cb, X_test_cb), (X_lgbm, X_test_lgbm), (X_xgb, X_test_xgb)


# ==========================================
# OPTUNA
# ==========================================
def tune_catboost(X_cb, y, kf):

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1000),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_seed': 42,
            'verbose': 0
        }

        scores = []
        for tr, va in kf.split(X_cb):
            model = CatBoostRegressor(**params, cat_features=SETTINGS["CAT_COLS"])
            model.fit(X_cb.iloc[tr], y.iloc[tr],
                      eval_set=(X_cb.iloc[va], y.iloc[va]),
                      early_stopping_rounds=50,
                      verbose=0)

            preds = model.predict(X_cb.iloc[va])
            scores.append(root_mean_squared_error(y.iloc[va], np.clip(preds, 0, None)))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best = study.best_params
    best['verbose'] = 0
    return best


# ==========================================
# TRAINING
# ==========================================
def train_models(X, y, X_test):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    (X_cb, X_test_cb), (X_lgbm, X_test_lgbm), (X_xgb, X_test_xgb) = build_model_inputs(X, X_test)

    best_cb = tune_catboost(X_cb, y, kf)

    preds = {
        "cb": np.zeros(len(X_test)),
        "lgbm": np.zeros(len(X_test)),
        "xgb": np.zeros(len(X_test))
    }

    for fold, (tr, va) in enumerate(kf.split(X)):
        print(f"🚀 Fold {fold+1}")

        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # CatBoost
        cb = CatBoostRegressor(**best_cb, cat_features=SETTINGS["CAT_COLS"])
        cb.fit(X_cb.iloc[tr], y_tr, verbose=0)
        preds["cb"] += np.clip(cb.predict(X_test_cb), 0, None) / 5

        # LightGBM
        lgbm = LGBMRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=7,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1
        )
        lgbm.fit(X_lgbm.iloc[tr], y_tr, categorical_feature=SETTINGS["CAT_COLS"])
        preds["lgbm"] += np.clip(lgbm.predict(X_test_lgbm), 0, None) / 5

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        xgb.fit(X_xgb.iloc[tr], y_tr,
                eval_set=[(X_xgb.iloc[va], y_va)], verbose=0)

        preds["xgb"] += np.clip(xgb.predict(X_test_xgb), 0, None) / 5

    return preds


# ==========================================
# FINAL PIPELINE
# ==========================================
def execute():

    print("=== PIPELINE START ===")

    id_map, train, features = read_inputs()
    format_dates(id_map, train, features)

    train_df, test_df = join_data(train, features, id_map)
    X, y, X_test = prepare_features(train_df, test_df)

    preds = train_models(X, y, X_test)

    print("📊 Combining predictions...")
    final = (preds["cb"] * 0.45) + (preds["lgbm"] * 0.40) + (preds["xgb"] * 0.15)

    submission = pd.DataFrame({
        "id": id_map["id"],
        "acons": final
    })

    submission.to_csv("/kaggle/working/submission_ensemble.csv", index=False)
    print("Submission saved!")


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    execute()
