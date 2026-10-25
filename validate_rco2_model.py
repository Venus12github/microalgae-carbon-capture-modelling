import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------
# Load Data
# -----------------------

df = pd.read_excel("prediction_data.xlsx")

# -----------------------
# Encode Species
# -----------------------

le = LabelEncoder()
df["Species_Code"] = le.fit_transform(df["Species"])

# -----------------------
# Inputs
# -----------------------

X = df[["Species_Code","CO2","NaHCO3"]]

# -----------------------
# Output
# -----------------------

y = df["RCO2 Mean"]

# -----------------------
# Train Model
# -----------------------

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

model.fit(X,y)

# -----------------------
# Predictions
# -----------------------

pred = model.predict(X)

# -----------------------
# R2
# -----------------------

r2 = r2_score(y,pred)

print("\nR² =",round(r2,4))

# -----------------------
# Table
# -----------------------

results = pd.DataFrame({
    "Observed":y,
    "Predicted":pred
})

print("\n")
print(results)

# -----------------------
# Plot
# -----------------------

plt.figure(figsize=(6,6))

plt.scatter(y,pred)

plt.plot(
    [y.min(),y.max()],
    [y.min(),y.max()]
)

plt.xlabel("Observed RCO2")
plt.ylabel("Predicted RCO2")

plt.title(
    f"Observed vs Predicted RCO2\nR²={r2:.3f}"
)

plt.savefig(
    "Observed_vs_Predicted_RCO2.png",
    dpi=300
)

plt.show()