import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Step 1: Load both files
friday = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
wednesday = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv')

friday.columns = friday.columns.str.strip()
wednesday.columns = wednesday.columns.str.strip()


# Step 2: Filter required attack types
friday_filtered = friday[friday['Label'].isin(['DDoS', 'BENIGN'])] # Friday: keep DDoS and BENIGN only


wed_attacks = ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'] # Wednesday: keep 4 attack types and BENIGN (excluding Heartbleed — only 11 samples, too few)
wednesday_filtered = wednesday[wednesday['Label'].isin(wed_attacks + ['BENIGN'])]

df = pd.concat([friday_filtered, wednesday_filtered], ignore_index=True)# Merge both datasets

# Binary label: 0 = normal traffic, 1 = attack
df['label_binary'] = (df['Label'] != 'BENIGN').astype(int)

print("=== Merged dataset distribution ===")
print(df['Label'].value_counts())
print(f"\nTotal samples: {len(df)}")
print(f"Attack: {df['label_binary'].sum()}")
print(f"Normal: {(df['label_binary']==0).sum()}")


# Step 3: Feature selection
# These 5 features capture DDoS behavior:
# - Slowloris: long duration, small packets, slow interval
# - DDoS/Hulk: high packet rate, large volume
features = [
    'Flow Duration',           # Connection duration
    'Average Packet Size',     # Avg packet size
    'Flow IAT Mean',           # Mean inter-arrival time
    'Fwd Packet Length Mean',  # Mean forward packet length
    'Flow Packets/s'           # Packets per second
]

X = df[features].copy()
y = df['label_binary']

X.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinite values and NaN
X.fillna(X.median(), inplace=True)

# Step 4: Normalize + train/test split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Step 5: Build ANN model
# Architecture: 5 -> 16 -> 8 -> 1
# Input:  5 network flow features
# Hidden: 2 layers with ReLU activation
# Output: 1 neuron with sigmoid (binary classification)
model = keras.Sequential([
    keras.layers.Input(shape=(5,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Step 6: Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

# Step 7: Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n=== Results ===")
print(classification_report(y_test, y_pred,
      target_names=['BENIGN', 'DDoS Attack']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training accuracy curve
axes[0].plot(history.history['accuracy'], label='train')
axes[0].plot(history.history['val_accuracy'], label='validation')
axes[0].set_title('Training Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
            xticklabels=['BENIGN', 'DDoS'],
            yticklabels=['BENIGN', 'DDoS'])
axes[1].set_title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Step 9: Export weights for ARM CPU
# Weights are quantized to integers using Q8 fixed-point format
# Multiply by 256 to preserve 2 decimal places of precision
# These integer arrays will be loaded into DM[] on the ARM CPU
print("\n=== C arrays for ARM CPU ===\n")

weights = model.get_weights()
SCALE = 256  # Q8 quantization

for i, w in enumerate(weights):
    flat = (w.flatten() * SCALE).astype(int)
    name = f"w{i}" if i % 2 == 0 else f"b{i//2}"
    print(f"int {name}[] = {{")
    print("  " + ", ".join(map(str, flat)))
    print("};\n")

# Scaler parameters needed for inference on ARM CPU
# Input features must be normalized before feeding into ANN
print("// Scaler mean (x256) — subtract from raw feature:")
mean_scaled = (scaler.mean_ * SCALE).astype(int)
print(f"int scaler_mean[] = {{{', '.join(map(str, mean_scaled))}}};\n")

print("// Scaler std (x256) — divide after subtracting mean:")
std_scaled = (scaler.scale_ * SCALE).astype(int)
print(f"int scaler_std[] = {{{', '.join(map(str, std_scaled))}}};\n")