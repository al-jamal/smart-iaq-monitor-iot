import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load a public regression dataset (fallback to synthetic if offline)
try:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df[["MedInc", "HouseAge", "AveRooms", "AveOccup"]].dropna()
    y = df.loc[X.index, "MedHouseVal"]
    dataset_name = "California Housing (Public Dataset)"
except Exception:
    rng = np.random.default_rng(42)
    n = 600
    temp = rng.uniform(18, 32, n)
    humidity = rng.uniform(30, 80, n)
    gas_ppm = rng.uniform(350, 1200, n)
    # Synthetic continuous comfort/IAQ score for Phase 1 starter test
    score = (
        100
        - 0.8 * np.abs(temp - 23)
        - 0.25 * np.abs(humidity - 50)
        - 0.03 * (gas_ppm - 400)
        + rng.normal(0, 2, n)
    )
    X = pd.DataFrame({"temp_c": temp, "humidity_pct": humidity, "gas_ppm": gas_ppm})
    y = pd.Series(score, name="iaq_score")
    dataset_name = "Synthetic Fallback Dataset"

print(f"Using dataset: {dataset_name} | Rows: {len(X)}")

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Intercept: {model.intercept_:.4f}")
print("Coefficients:", dict(zip(X.columns, np.round(model.coef_, 4))))
print(f"Test MSE: {mse:.4f}")
print(f"Test R^2: {r2:.4f}")
