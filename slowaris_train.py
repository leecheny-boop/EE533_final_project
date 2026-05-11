import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv')

df.columns = df.columns.str.strip()

df = df[df['Label'].isin(['BENIGN', 'DoS slowloris'])]

print("data numbers：")
print(df['Label'].value_counts())


features = [
    'Flow Duration',        # 連線持續時間
    'Average Packet Size',  # 平均封包大小
    'Flow IAT Mean',        # 封包間隔平均
    'Fwd Packet Length Mean', # 前向封包長度
    'Flow Packets/s'        # 每秒封包數
]

X = df[features].copy()
y = (df['Label'] == 'DoS slowloris').astype(int)  # 1=attack, 0=normal

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print("\n特徵統計：")
print(X.describe())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\ntrain：{len(X_train)} ，test：{len(X_test)} ")

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(5,)),  # 隱藏層 1
    keras.layers.Dense(8,  activation='relu'),                     # 隱藏層 2
    keras.layers.Dense(1,  activation='sigmoid')                   # 輸出層
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n=== 結果 ===")
print(classification_report(y_test, y_pred, target_names=['BENIGN', 'Slowloris']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n=== ARM CPU 用的 C 陣列 ===\n")

weights = model.get_weights()

SCALE = 256  

for i, w in enumerate(weights):
    flat = (w.flatten() * SCALE).astype(int)
    name = f"w{i}" if i % 2 == 0 else f"b{i//2}"
    print(f"int {name}[] = {{")
    print("  " + ", ".join(map(str, flat)))
    print("};\n")

print("// Scaler mean (乘以256):")
mean_scaled = (scaler.mean_ * SCALE).astype(int)
print(f"int scaler_mean[] = {{{', '.join(map(str, mean_scaled))}}};\n")

print("// Scaler std (乘以256):")
std_scaled = (scaler.scale_ * SCALE).astype(int)
print(f"int scaler_std[] = {{{', '.join(map(str, std_scaled))}}};\n")