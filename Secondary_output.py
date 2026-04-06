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
# CONFIG SERVICE
# ==========================================
class CFG:
    INPUT = "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge"
    FEATURES = "/kaggle/input/datasets/racharladhanush/catched-features/cached_features.parquet"
    IDMAP = "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge/id_mapping_new.csv"

    CAT = ['vehicle', 'shift', 'operator_id']
    NUM = [
        'mean_speed','max_speed','mean_altitude','std_altitude',
        'max_altitude','min_altitude','idle_pings','total_pings',
        'min_cumdist','max_cumdist','distance_travelled','net_lift'
    ]


# ==========================================
# DATA SERVICE
# ==========================================
class DataService:

    @staticmethod
    def fetch():
        id_map = pd.read_csv(CFG.IDMAP)

        files = glob.glob(os.path.join(CFG.INPUT, "smry_*_train_ordered.csv"))
        if not files:
            files = glob.glob(os.path.join(CFG.INPUT, "*", "smry_*_train_ordered.csv"))

        train = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        feats = pd.read_parquet(CFG.FEATURES)

        for df in [id_map, train, feats]:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        return id_map, train, feats


# ==========================================
# FEATURE SERVICE
# ==========================================
class FeatureService:

    @staticmethod
    def prepare(id_map, train, feats):
        train_df = train.merge(feats, on=['vehicle','date','shift'], how='left')
        test_df = id_map.merge(feats, on=['vehicle','date','shift'], how='left')

        for c in CFG.CAT:
            train_df[c] = train_df[c].fillna("UNKNOWN").astype(str)
            test_df[c] = test_df[c].fillna("UNKNOWN").astype(str)

        train_df = train_df.dropna(subset=['acons']).reset_index(drop=True)

        X = train_df[CFG.CAT + CFG.NUM]
        y = train_df['acons']
        X_test = test_df[CFG.CAT + CFG.NUM]

        return train_df, X, y, X_test


# ==========================================
# ENCODING SERVICE
# ==========================================
class EncodingService:

    @staticmethod
    def transform(X, X_test):

        # CatBoost
        X_cb, X_test_cb = X.copy(), X_test.copy()

        # LightGBM
        X_lgbm, X_test_lgbm = X.copy(), X_test.copy()
        for c in CFG.CAT:
            X_lgbm[c] = X_lgbm[c].astype('category')
            X_test_lgbm[c] = pd.Categorical(X_test_lgbm[c], categories=X_lgbm[c].cat.categories)

        # XGBoost
        X_xgb, X_test_xgb = X_lgbm.copy(), X_test_lgbm.copy()
        for c in CFG.CAT:
            X_xgb[c] = X_xgb[c].cat.codes
            X_test_xgb[c] = X_test_xgb[c].cat.codes

        return (X_cb, X_test_cb), (X_lgbm, X_test_lgbm), (X_xgb, X_test_xgb)


# ==========================================
# MODEL SERVICE
# ==========================================
class ModelService:

    def __init__(self):
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def tune_catboost(self, X_cb, y):

        def objective(trial):
            params = {
                'iterations': trial.suggest_int(500, 1000),
                'depth': trial.suggest_int(4, 8),
                'learning_rate': trial.suggest_float(0.01, 0.1, log=True),
                'subsample': trial.suggest_float(0.6, 1.0),
                'random_seed': 42,
                'verbose': 0
            }

            scores = []
            for tr, va in self.kf.split(X_cb):
                model = CatBoostRegressor(**params, cat_features=CFG.CAT)
                model.fit(X_cb.iloc[tr], y.iloc[tr],
                          eval_set=(X_cb.iloc[va], y.iloc[va]),
                          early_stopping_rounds=50,
                          verbose=0)

                preds = model.predict(X_cb.iloc[va])
                scores.append(root_mean_squared_error(y.iloc[va], np.clip(preds,0,None)))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        best = study.best_params
        best['verbose'] = 0
        return best

    def train(self, X, y, X_test):

        (X_cb, X_test_cb), (X_lgbm, X_test_lgbm), (X_xgb, X_test_xgb) = EncodingService.transform(X, X_test)

        best_cb = self.tune_catboost(X_cb, y)

        preds_test = {k: np.zeros(len(X_test)) for k in ['cb','lgbm','xgb']}
        preds_oof = {k: np.zeros(len(X)) for k in ['cb','lgbm','xgb']}

        for fold, (tr, va) in enumerate(self.kf.split(X)):
            print(f"Fold {fold+1}")

            y_tr, y_va = y.iloc[tr], y.iloc[va]

            # CatBoost
            cb = CatBoostRegressor(**best_cb, cat_features=CFG.CAT)
            cb.fit(X_cb.iloc[tr], y_tr, verbose=0)
            preds_oof['cb'][va] = np.clip(cb.predict(X_cb.iloc[va]),0,None)
            preds_test['cb'] += np.clip(cb.predict(X_test_cb),0,None)/5

            # LightGBM
            lgbm = LGBMRegressor(n_estimators=600, learning_rate=0.03,
                                 max_depth=7, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbosity=-1)
            lgbm.fit(X_lgbm.iloc[tr], y_tr, categorical_feature=CFG.CAT)
            preds_oof['lgbm'][va] = np.clip(lgbm.predict(X_lgbm.iloc[va]),0,None)
            preds_test['lgbm'] += np.clip(lgbm.predict(X_test_lgbm),0,None)/5

            # XGB
            xgb = XGBRegressor(n_estimators=600, learning_rate=0.03,
                               max_depth=6, subsample=0.8,
                               colsample_bytree=0.8, random_state=42)
            xgb.fit(X_xgb.iloc[tr], y_tr,
                    eval_set=[(X_xgb.iloc[va], y_va)], verbose=0)

            preds_oof['xgb'][va] = np.clip(xgb.predict(X_xgb.iloc[va]),0,None)
            preds_test['xgb'] += np.clip(xgb.predict(X_test_xgb),0,None)/5

        return preds_oof, preds_test


# ==========================================
# REPORT SERVICE
# ==========================================
class ReportService:

    @staticmethod
    def generate(train_df):

        # Route Benchmark
        route_cols = ['distance_travelled','net_lift','mean_altitude',
                      'std_altitude','max_altitude','min_altitude']

        model = LGBMRegressor(n_estimators=150, max_depth=5)
        model.fit(train_df[route_cols].fillna(0), train_df['acons'])

        train_df['expected_fuel_independent_of_dumper'] = model.predict(train_df[route_cols].fillna(0))
        train_df['fuel_wasted'] = train_df['acons'] - train_df['expected_fuel_independent_of_dumper']

        # Save outputs
        train_df.to_csv("/kaggle/working/benchmark_full.csv", index=False)

        dumper = train_df.groupby('vehicle')['fuel_wasted'].mean().reset_index()
        dumper.to_csv("/kaggle/working/dumper_eff.csv", index=False)

        operator = train_df.groupby('operator_id')['fuel_wasted'].mean().reset_index()
        operator.to_csv("/kaggle/working/operator_eff.csv", index=False)


# ==========================================
# ORCHESTRATOR
# ==========================================
class Pipeline:

    def run(self):
        print("🚀 Starting Full Pipeline")

        id_map, train, feats = DataService.fetch()
        train_df, X, y, X_test = FeatureService.prepare(id_map, train, feats)

        model = ModelService()
        oof, test = model.train(X, y, X_test)

        weights = {'cb':0.45,'lgbm':0.40,'xgb':0.15}

        final_test = sum(test[k]*weights[k] for k in weights)
        final_train = sum(oof[k]*weights[k] for k in weights)

        train_df['pred_acons'] = final_train

        # Save submission
        pd.DataFrame({
            "id": id_map['id'],
            "acons": final_test
        }).to_csv("/kaggle/working/submission_ensemble.csv", index=False)

        # Reports
        ReportService.generate(train_df)

        print("Everything completed!")


# ==========================================
if __name__ == "__main__":
    Pipeline().run()
