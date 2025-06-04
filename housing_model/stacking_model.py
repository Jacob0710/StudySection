# -*- coding: utf-8 -*-
"""
房價預測 Stacking 模型
本程式以中文撰寫，並以 '----' 作為 cell 分隔符號。
"""

# ----
# 📌 Cell 1：導入模組與定義評估函式

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

from sklearn.linear_model import LinearRegression


def evaluate_model(y_true, y_pred):
    """回傳 R²、RMSE 及 MAPE 指標"""
    return {
        "R²": round(r2_score(y_true, y_pred), 4),
        "RMSE": f"{np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}",
        "MAPE": f"{mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}%"
    }

# ----
# 📌 Cell 2：資料讀取與標準化處理

file_path = r"D:/研究生活/房價資料/102-112房屋資料-CLEAN/FinalData.csv"
df = pd.read_csv(file_path)

lower = df['total_price'].quantile(0.05)
upper = df['total_price'].quantile(0.95)
df = df[(df['total_price'] >= lower) & (df['total_price'] <= upper)]

df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df = df.sort_values(by='transaction_date').reset_index(drop=True)

df = df.drop(columns=['address'])

df_raw = df.copy()

df = df.apply(pd.to_numeric, errors='coerce')
scaler = RobustScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

df.index = range(1, len(df) + 1)

# ----
# 📌 Cell 3：定義特徵與目標變數

features = [
    "building_shifting_total_area", "latitude", "longitude",
    "shifting_total_area", "transaction_date", "complete_date",
    "houseAge", "shifting_floor", "district"
]

X = df[features]
y = df['total_price'].values
y_orig = df_raw['total_price'].values

col_idx = list(df.columns).index("total_price")
median_tp = scaler.center_[col_idx]
iqr_tp = scaler.scale_[col_idx]

n_samples = X.shape[0]

# ----
# 📌 Cell 4：設定模型超參數與交叉驗證

tscv = TimeSeriesSplit(n_splits=10)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 81,
    'min_data_in_leaf': 9,
    'max_depth': 8,
    'learning_rate': 0.0607,
    'feature_fraction': 0.6937,
    'bagging_fraction': 0.7819,
    'bagging_freq': 3,
    'lambda_l1': 9.6594e-05,
    'lambda_l2': 0.03406,
    'n_estimators': 693,
    'verbosity': -1
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'max_depth': 9,
    'learning_rate': 0.0313,
    'subsample': 0.8187,
    'colsample_bytree': 0.7328,
    'min_child_weight': 5,
    'gamma': 0.002,
    'reg_alpha': 0.0494,
    'reg_lambda': 0.0117,
    'n_estimators': 697
}

n_units1 = 512
n_units2 = 192
lr = 0.004892
recur_drop = 0.4005
spatial_drop = 0.09097
dense_units = 96
drop_rate = 0.1828
epochs = 10
batch_size = 32

# ----
# 📌 Cell 5：執行 LightGBM OOF 預測

oof_lgb = np.zeros(n_samples)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    model = LGBMRegressor(**lgb_params)
    model.fit(X.iloc[train_idx], y[train_idx])
    y_pred_scaled = model.predict(X.iloc[val_idx])
    y_pred = y_pred_scaled * iqr_tp + median_tp
    oof_lgb[val_idx] = y_pred
    print(f"LightGBM Fold {fold}: {evaluate_model(y_orig[val_idx], y_pred)}")

# ----
# 📌 Cell 6：執行 XGBoost OOF 預測

oof_xgb = np.zeros(n_samples)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    model = XGBRegressor(**xgb_params)
    model.fit(X.iloc[train_idx], y[train_idx], eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
    y_pred_scaled = model.predict(X.iloc[val_idx])
    y_pred = y_pred_scaled * iqr_tp + median_tp
    oof_xgb[val_idx] = y_pred
    print(f"XGBoost Fold {fold}: {evaluate_model(y_orig[val_idx], y_pred)}")

# ----
# 📌 Cell 7：執行 LSTM OOF 預測

oof_lstm = np.zeros(n_samples)
n_features = X.shape[1]

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    X_train = X.iloc[train_idx].values.reshape(-1, 1, n_features)
    X_val = X.iloc[val_idx].values.reshape(-1, 1, n_features)
    y_train = y[train_idx]
    y_val = y[val_idx]

    model = Sequential()
    model.add(SpatialDropout1D(spatial_drop, input_shape=(1, n_features)))
    model.add(LSTM(n_units1, return_sequences=True, recurrent_dropout=recur_drop))
    model.add(Dropout(drop_rate))
    model.add(LSTM(n_units2, recurrent_dropout=recur_drop))
    model.add(Dropout(drop_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred_scaled = model.predict(X_val).flatten()
    y_pred = y_pred_scaled * iqr_tp + median_tp
    oof_lstm[val_idx] = y_pred
    print(f"LSTM Fold {fold}: {evaluate_model(y_orig[val_idx], y_pred)}")

# ----
# 📌 Cell 8：建立 Stacking 模型與總體預測

meta_X = np.vstack((oof_lgb, oof_xgb, oof_lstm)).T
meta_y = y_orig

meta_model = LinearRegression()
meta_model.fit(meta_X, meta_y)

final_pred = meta_model.predict(meta_X)
result = evaluate_model(meta_y, final_pred)
print(f"Stacking Model Performance：R²={result['R²']}, RMSE={result['RMSE']}, MAPE={result['MAPE']}")

# ----
