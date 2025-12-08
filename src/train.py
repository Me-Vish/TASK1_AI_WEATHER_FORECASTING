from data_preprocessing import load_data, split_xy
from model import build_model
import joblib

# 1. Load dataset
df = load_data("Data/seattle_weather.csv")

# 2. Choose columns
feature = "temp_max"      # input
target = "temp_min"       # output prediction

X, y = split_xy(df, feature, target)

# 3. Build model
model = build_model()

# 4. Train model
model.fit(X, y, epochs=20, batch_size=16)

# 5. Save model
model.save("models/weather_simple_model.h5")

print("Model training complete! ✔ Saved in models folder.")
