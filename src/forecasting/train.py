import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ─── CONFIG ────────────────────────────────────────────────
DATA_PATH = "data/raw/DataCoSupplyChainDataset.csv"
PROCESSED_PATH = "data/processed/"
SEQUENCE_LENGTH = 30   # Use past 30 days to predict next day
TEST_SPLIT = 0.2

os.makedirs(PROCESSED_PATH, exist_ok=True)

# ─── STEP 1: LOAD DATA ─────────────────────────────────────
print("📦 Loading data...")
df = pd.read_csv(DATA_PATH, encoding='latin-1')
print(f"   Raw shape: {df.shape}")

# ─── STEP 2: SELECT & CLEAN ────────────────────────────────
print("🧹 Cleaning data...")

df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])

# Aggregate daily demand
daily = df.groupby('order date (DateOrders)').agg(
    quantity    = ('Order Item Quantity', 'sum'),
    revenue     = ('Order Item Total', 'sum'),
    late_risk   = ('Late_delivery_risk', 'mean'),
    num_orders  = ('Order Item Quantity', 'count')
).reset_index()

daily.columns = ['date', 'quantity', 'revenue', 'late_risk', 'num_orders']
daily = daily.sort_values('date').reset_index(drop=True)

print(f"   Date range: {daily['date'].min()} → {daily['date'].max()}")
print(f"   Total days: {len(daily)}")

# ─── STEP 3: FEATURE ENGINEERING ───────────────────────────
print("🔧 Engineering features...")

# Time features
daily['day_of_week']  = daily['date'].dt.dayofweek
daily['month']        = daily['date'].dt.month
daily['quarter']      = daily['date'].dt.quarter
daily['is_weekend']   = (daily['day_of_week'] >= 5).astype(int)

# Lag features (past values as inputs)
for lag in [1, 2, 3, 7, 14]:
    daily[f'quantity_lag_{lag}'] = daily['quantity'].shift(lag)

# Rolling statistics
daily['rolling_mean_7']  = daily['quantity'].rolling(window=7).mean()
daily['rolling_std_7']   = daily['quantity'].rolling(window=7).std()
daily['rolling_mean_14'] = daily['quantity'].rolling(window=14).mean()

# Drop rows with NaN (from lag/rolling)
daily = daily.dropna().reset_index(drop=True)
print(f"   Shape after feature engineering: {daily.shape}")

# ─── STEP 4: SCALE ─────────────────────────────────────────
print("📏 Scaling features...")

feature_cols = [
    'quantity', 'revenue', 'late_risk', 'num_orders',
    'day_of_week', 'month', 'quarter', 'is_weekend',
    'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3',
    'quantity_lag_7', 'quantity_lag_14',
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14'
]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(daily[feature_cols])
scaled_df = pd.DataFrame(scaled, columns=feature_cols)

# Save scaler for later use in prediction
with open(f"{PROCESSED_PATH}scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Scaler saved")

# ─── STEP 5: CREATE SEQUENCES ──────────────────────────────
print("🔄 Creating sequences for LSTM...")

def create_sequences(data, target_col_idx, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])         # Past seq_length days
        y.append(data[i, target_col_idx])       # Next day quantity
    return np.array(X), np.array(y)

target_idx = feature_cols.index('quantity')
X, y = create_sequences(scaled, target_idx, SEQUENCE_LENGTH)

print(f"   X shape: {X.shape}  →  (samples, timesteps, features)")
print(f"   y shape: {y.shape}  →  (samples,)")

# ─── STEP 6: TRAIN/TEST SPLIT ──────────────────────────────
split = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n   Train samples: {len(X_train)}")
print(f"   Test samples:  {len(X_test)}")

# ─── STEP 7: SAVE PROCESSED DATA ───────────────────────────
np.save(f"{PROCESSED_PATH}X_train.npy", X_train)
np.save(f"{PROCESSED_PATH}X_test.npy",  X_test)
np.save(f"{PROCESSED_PATH}y_train.npy", y_train)
np.save(f"{PROCESSED_PATH}y_test.npy",  y_test)

daily.to_csv(f"{PROCESSED_PATH}daily_features.csv", index=False)

print("\n✅ All processed data saved to data/processed/")
print("   → X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("   → scaler.pkl")
print("   → daily_features.csv")
print(f"\n📐 Input shape for LSTM: {X_train.shape}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ─── MODEL CONFIG ──────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 32
MODEL_PATH = "models/lstm_demand_model/"
os.makedirs(MODEL_PATH, exist_ok=True)

n_features  = X_train.shape[2]   # Number of input features
n_timesteps = X_train.shape[1]   # Sequence length

print(f"\n🧠 Building model...")
print(f"   Input: ({n_timesteps} timesteps, {n_features} features)")

# ─── BUILD MODEL ───────────────────────────────────────────
model = Sequential([

    # 1D-CNN layer to detect local patterns in time-series
    Conv1D(filters=64, kernel_size=3, activation='relu',
           input_shape=(n_timesteps, n_features)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # LSTM layer to learn long-term dependencies
    LSTM(128, return_sequences=True),
    Dropout(0.2),

    LSTM(64, return_sequences=False),
    Dropout(0.2),

    # Output layer
    Dense(32, activation='relu'),
    Dense(1)   # Predict 1 value: next day demand
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ─── CALLBACKS ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=f"{MODEL_PATH}best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ─── TRAIN ─────────────────────────────────────────────────
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUATE ──────────────────────────────────────────────
print("\n📊 Evaluating on test set...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss (MSE): {test_loss:.4f}")
print(f"   Test MAE:        {test_mae:.4f}")

# ─── PLOT TRAINING HISTORY ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Model Loss (MSE)')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_title('Model MAE')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{MODEL_PATH}training_history.png")
plt.show()
print(f"   ✅ Training chart saved")

# ─── SAVE FINAL MODEL ──────────────────────────────────────
model.save(f"{MODEL_PATH}final_model.keras")
print(f"\n✅ Model saved to {MODEL_PATH}")
print("   → best_model.keras")
print("   → final_model.keras")
print("   → training_history.png")