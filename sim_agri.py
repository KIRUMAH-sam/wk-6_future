# simulate_and_train_agri.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# generate N fields
N = 500
days = 120  # growing season
def gen_field(seed):
    np.random.seed(seed)
    # daily temperature around 20-30 with seasonal trend
    t = 20 + 5*np.sin(np.linspace(0,3.14,days)) + np.random.randn(days)*1.5
    # daily cumulative rain random bursts
    rain = np.random.poisson(0.4, size=days) * np.random.uniform(0,10,size=days)
    # soil moisture responds to rain + evapotranspiration
    sm = np.clip(0.2 + 0.003*np.cumsum(rain) - 0.001*np.arange(days) + np.random.randn(days)*0.02, 0, 1)
    # NDVI grows then saturates
    ndvi = np.clip(0.2 + 0.6*(1 - np.exp(-np.linspace(0,3,days))) + 0.05*np.random.randn(days), -1,1)
    # features: mean temp, total rain, mean soil moisture, ndvi slope
    feat = {
        "mean_temp": t.mean(),
        "total_rain": rain.sum(),
        "mean_sm": sm.mean(),
        "ndvi_slope": (ndvi[-1]-ndvi[0])/days
    }
    # yield depends on rain (middle), good moisture, ndvi slope
    yield_val = 1000 + 50*feat["mean_temp"] + 2*feat["total_rain"] + 800*feat["mean_sm"] + 1200*feat["ndvi_slope"] + np.random.randn()*300
    return feat, yield_val

rows=[]
y=[]
for i in range(N):
    f, val = gen_field(i)
    rows.append(f)
    y.append(val)

df = pd.DataFrame(rows)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
# Save model (sklearn)
import joblib
joblib.dump(model, "agri_yield_rf.joblib")
print("Model saved.")
