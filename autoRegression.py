from preprocessing import *
from scipy.stats import describe
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(y_true, y_pred, name="Validation"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"=== {name} Metrics ===")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2:   {r2:.4f}\n")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

data_path = "used_cars.csv"
auto_data = pd.read_csv(data_path)

# 1. Очистка численных колонок
auto_data[['price', 'milage']] = auto_data[['price', 'milage']].apply(drop_num_artifacts)

# 2. Очистка целевой переменной от экстремальных выбросов
auto_data = auto_data[(auto_data['price'] >= 500) & (auto_data['price'] <= 300000)].reset_index(drop=True)

auto_data['accident'] = convert_and_fillna_mode(
    auto_data['accident'],
    {"At least 1 accident or damage reported": 1, "None reported": 0}
)
auto_data['car_age']=2026-auto_data['model_year']
auto_data.drop(columns=['clean_title'],inplace=True)
auto_data.drop(columns=['model_year'],inplace=True)

y = auto_data.price
X = auto_data.drop(['price', 'model'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Препроцессинг
X_train, X_valid = process_transmission_column(X_train, X_valid, 'transmission')
X_train, X_valid = process_engine_column(X_train, X_valid, 'engine')
X_train, X_valid = target_encode_with_smoothing(X_train, X_valid, y_train, "brand", 10)
X_train, X_valid = target_encode_with_smoothing(X_train, X_valid, y_train, "ext_col", 10)
X_train, X_valid = target_encode_with_smoothing(X_train, X_valid, y_train, "int_col", 10)
X_train, X_valid = process_fuel_type_column(X_train, X_valid, "fuel_type")

print(X_train.head(40))

# Масштабирование через RobustScaler (устойчив к выбросам)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Логарифмирование целевой переменной
y_train_log = np.log1p(y_train)

# Автоматический подбор оптимального коэффициента регуляризации
model = RidgeCV(alphas=np.logspace(-2, 3, 20))
model.fit(X_train_scaled, y_train_log)

# Ограничиваем отрицательные предсказания минимально возможной ценой
min_log_price = np.log1p(500)
y_train_pred_log = np.clip(model.predict(X_train_scaled), a_min=min_log_price, a_max=None)
y_valid_pred_log = np.clip(model.predict(X_valid_scaled), a_min=min_log_price, a_max=None)

y_train_pred = np.expm1(y_train_pred_log)
y_valid_pred = np.expm1(y_valid_pred_log)

evaluate_model(y_train, y_train_pred, "Train")
evaluate_model(y_valid, y_valid_pred, "Validation")

coefficients = pd.Series(model.coef_, index=X_train.columns).sort_values(ascending=False)

print("Топ-5 признаков, увеличивающих цену:")
print(coefficients.head(5))

print("\nТоп-5 признаков, уменьшающих цену:")
print(coefficients.tail(5))