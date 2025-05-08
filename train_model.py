
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# ✅ Replace this with your actual dataset filename
df = pd.read_csv("employee_data.csv")  # <-- CHANGE THIS

# X = features, y = target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "attrition_model.pkl")
print("✅ Model saved as attrition_model.pkl")
