import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Initializing AeroPredict-ML Training Sequence...")

# ==========================================
# 1. LOAD THE DATA
# ==========================================
columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
          [f's{i}' for i in range(1, 22)]

# Safely find the data folder from inside the src/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
file_path = os.path.join(project_root, 'data', 'train_FD001.txt')

print("Loading NASA C-MAPSS data...")
df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

# ==========================================
# 2. CALCULATE PIECEWISE RUL (Capped at 125)
# ==========================================
max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True)
df = df.merge(max_cycles, on='engine_id', how='left')

df['RUL'] = df['max_cycle'] - df['cycle']
df['RUL'] = df['RUL'].clip(upper=125) # The Aerospace Logic Trick
df.drop('max_cycle', axis=1, inplace=True)

# ==========================================
# 3. CLEAN THE DATA
# ==========================================
sensor_cols = [f's{i}' for i in range(1, 22)]
sensor_stdevs = df[sensor_cols].std()
flat_sensors = sensor_stdevs[sensor_stdevs < 0.0001].index.tolist()

df_cleaned = df.drop(columns=flat_sensors)
print(f"Dropped static sensors to reduce noise: {flat_sensors}")

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
X = df_cleaned.drop(columns=['engine_id', 'cycle', 'RUL'])
y = df_cleaned['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Random Forest on {X_train.shape[0]} samples...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATE
# ==========================================
predictions = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n==========================================")
print("     FINAL MODEL PERFORMANCE METRICS      ")
print("==========================================")
print(f"Mean Absolute Error (MAE):       {mae:.2f} cycles")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} cycles")
print("==========================================")
print("Training Complete. Model is ready for deployment.")