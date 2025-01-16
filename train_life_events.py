import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from app.utils import create_sequences, save_scaler


sys.stdout.reconfigure(encoding='utf-8')
# Load dataset with utf-8 encoding and handle bad lines
data = pd.read_csv(
    "data/financial_data.csv",
    encoding='utf-8',
    on_bad_lines='skip'  # Skips lines with bad formatting
)

# Check for special characters and clean if necessary
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # Remove extra spaces

# Convert Date to numeric features if needed
if 'Date' in data.columns:
    data['Year'] = pd.to_datetime(data['Date'], errors='coerce').dt.year
    data['Month'] = pd.to_datetime(data['Date'], errors='coerce').dt.month
    data['Day'] = pd.to_datetime(data['Date'], errors='coerce').dt.day
    data = data.drop(columns=['Date'])

# Drop rows with any NaN values after processing
data = data.dropna()

# One-hot encode categorical columns
categorical_columns = ['Transaction_Type', 'Occupation', 'Location']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Ensure Life_Event is cleaned and encoded
data = data[data['Life_Event'] != '-']
label_encoder = LabelEncoder()
data['Life_Event'] = label_encoder.fit_transform(data['Life_Event'])

# Define sequence length
seq_length = 3  # Adjust to your dataset size
X, y = create_sequences(data, seq_length, target_col="Life_Event")

# Ensure enough data for train-test split
if len(X) <= 1:
    raise ValueError("Dataset too small to generate sufficient sequences. Please reduce sequence length or add more data.")

# Adjust test_size based on the number of samples
test_size = 0.5 if len(X) < 10 else 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Normalize numerical features
numerical_columns = ["Transaction_Amount", "Savings", "Monthly_Spending", "Cumulative_Savings"]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape).astype('float32')
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape).astype('float32')

# Save the scaler for later use
save_scaler(scaler, "app/models/scaler.pkl")

# Define and compile the LSTM model
model = Sequential([
    Input(shape=(seq_length, X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save the model
model.save("app/models/life_event_model.h5")
print("Model trained and saved successfully.")
