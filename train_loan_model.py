import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import sys


sys.stdout.reconfigure(encoding='utf-8')

# Configure logging with UTF-8 encoding
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('app.log', encoding='utf-8')])

# Load dataset with UTF-8 encoding to avoid Unicode errors
logging.info("Loading dataset...")
try:
    data = pd.read_csv("data/loan_recommendation_dataset.csv", encoding='utf-8-sig')
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

# Preview the data to ensure correct loading
logging.info(f"Data preview: {data.head()}")

# Handle categorical variables (one-hot encoding)
data = pd.get_dummies(data, columns=['loan_type', 'user_location'], drop_first=True)

# Normalize numerical columns (e.g., user_income, loan_amount)
scaler = StandardScaler()
data[['user_income', 'loan_amount']] = scaler.fit_transform(data[['user_income', 'loan_amount']])

# Separate features (X) and target (y)
X = data.drop(columns=['suitability_score'])  # All features except the target
y = data['suitability_score']  # Suitability score as the target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Data split completed. Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")

# Define the input layer for features
input_features = Input(shape=(X_train.shape[1],), name="input_features")

# Define hidden layers
hidden = Dense(64, activation='relu')(input_features)
hidden = Dense(32, activation='relu')(hidden)

# Output layer
output = Dense(1, activation='linear', name="output")(hidden)  # Linear activation for regression

# Compile the model
model = Model(inputs=input_features, outputs=output)
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

logging.info("Model defined and compiled successfully.")

# Train the model
logging.info("Starting model training...")
try:
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    logging.info("Model training completed successfully.")
except Exception as e:
    logging.error(f"Error during model training: {e}")
    raise

# Evaluate the model
logging.info("Evaluating model on test data...")
loss, mae = model.evaluate(X_test, y_test)
logging.info(f"Test loss: {loss}, Test MAE: {mae}")

# Save the model
logging.info("Saving the trained model...")
try:
    model.save("loan_recommendation_model.h5")
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Error saving the model: {e}")
    raise
