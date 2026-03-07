import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─── LOAD ARTIFACTS ────────────────────────────────────────
print("📦 Loading model and data...")

model  = tf.keras.models.load_model("models/lstm_demand_model/best_model.keras")

with open("data/processed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test  = np.load("data/processed/X_test.npy")
y_test  = np.load("data/processed/y_test.npy")

print(f"   Model loaded ✅")
print(f"   Test set: {X_test.shape}")

# ─── PREDICT ───────────────────────────────────────────────
print("\n🔮 Generating predictions...")
y_pred_scaled = model.predict(X_test)

# Inverse scale predictions back to original quantity values
# We need to create a dummy array to inverse transform only the quantity column
dummy = np.zeros((len(y_pred_scaled), scaler.n_features_in_))
dummy[:, 0] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy)[:, 0]

dummy2 = np.zeros((len(y_test), scaler.n_features_in_))
dummy2[:, 0] = y_test.flatten()
y_actual = scaler.inverse_transform(dummy2)[:, 0]

# ─── METRICS ───────────────────────────────────────────────
mae  = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100

print(f"\n📊 Forecast Metrics:")
print(f"   MAE:  {mae:.2f}  (avg units off per day)")
print(f"   RMSE: {rmse:.2f}")
print(f"   MAPE: {mape:.2f}%")

# ─── PLOT PREDICTIONS ──────────────────────────────────────
plt.figure(figsize=(15, 5))
plt.plot(y_actual[:100], label='Actual Demand',    color='steelblue',  linewidth=2)
plt.plot(y_pred[:100],   label='Predicted Demand', color='orangered',  linewidth=2, linestyle='--')
plt.title('LSTM Demand Forecast vs Actual (First 100 Test Days)', fontsize=14)
plt.xlabel('Days')
plt.ylabel('Units Ordered')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("models/lstm_demand_model/predictions.png")
plt.show()

print("\n✅ Prediction chart saved to models/lstm_demand_model/predictions.png")