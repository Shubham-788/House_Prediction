import joblib

# Load your existing model
model = joblib.load("model.pkl")  # If it's already a pickle file
scaler = joblib.load("scaler.pkl")
# Save with compression
joblib.dump(model, "model.pkl", compress=5)  # Adjust compression level as needed
joblib.dump(scaler, "scaler.pkl", compress=5)
