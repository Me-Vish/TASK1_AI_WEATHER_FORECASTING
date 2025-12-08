from tensorflow.keras.models import load_model
import pandas as pd

# Load trained model
model = load_model("models/weather_simple_model.h5")

# Example input (you can change this value)
new_value = [[50]]   # example max temp

result = model.predict(new_value)

print("Predicted value:", result)
