import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from structured import fix_data_frame

# Ensure 'models' directory exists
if not os.path.exists('models'):
    os.mkdir('models')

# Load and preprocess data
df = pd.read_csv("datasets/insurance.csv")
df = fix_data_frame(df)

# Separate features and target
x = df.drop(columns='age')
y = df['age']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_jobs=-1)
print('Training model...')
model.fit(x_train, y_train)

# Save the model
model_path = f'models/r{model.score(x_train, y_train)}-model.pkl'
joblib.dump(model, model_path)
print(f'Model trained and saved at: {model_path}')
