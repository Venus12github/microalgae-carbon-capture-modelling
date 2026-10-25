import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ==================================
# LOAD DATA
# ==================================

df = pd.read_excel("prediction_data_replicates.xlsx.xlsx")

# ==================================
# ENCODE SPECIES
# ==================================

le = LabelEncoder()

df["Species_Code"] = le.fit_transform(df["Species"])

# ==================================
# INPUTS AND OUTPUTS
# ==================================

X = df[["Species_Code", "CO₂", "NaHCO3"]]

y_prod = df["Productivity"]
y_rco2 = df["RCO2"]

# ==================================
# TRAIN MODELS
# ==================================

prod_model = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

rco2_model = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

prod_model.fit(X, y_prod)
rco2_model.fit(X, y_rco2)

# ==================================
# USER INPUT
# ==================================

species = input("Species (CS/SO): ").upper()

co2 = float(input("CO2 (%): "))

nahco3 = float(input("NaHCO3 (g/L): "))
prod_target = float(
    input("Productivity Target: ")
)

rco2_target = float(
    input("RCO2 Target: ")
)
# ==================================
# PREDICTION
# ==================================

species_code = le.transform([species])[0]

new_data = pd.DataFrame({
    "Species_Code": [species_code],
    "CO₂": [co2],
    "NaHCO3": [nahco3]
})

pred_prod = prod_model.predict(new_data)[0]
pred_rco2 = rco2_model.predict(new_data)[0]

# ==================================
# TREE-BASED UNCERTAINTY
# ==================================

prod_tree_preds = []

for tree in prod_model.estimators_:
    prod_tree_preds.append(
        tree.predict(new_data)[0]
    )

rco2_tree_preds = []

for tree in rco2_model.estimators_:
    rco2_tree_preds.append(
        tree.predict(new_data)[0]
    )

prod_tree_preds = np.array(prod_tree_preds)
rco2_tree_preds = np.array(rco2_tree_preds)

# ==================================
# PRODUCTIVITY STATS
# ==================================

prod_mean = np.mean(prod_tree_preds)
prod_sd = np.std(prod_tree_preds)

prod_p5 = np.percentile(prod_tree_preds, 5)
prod_p50 = np.percentile(prod_tree_preds, 50)
prod_p95 = np.percentile(prod_tree_preds, 95)

# ==================================
# RCO2 STATS
# ==================================

rco2_mean = np.mean(rco2_tree_preds)
rco2_sd = np.std(rco2_tree_preds)

rco2_p5 = np.percentile(rco2_tree_preds, 5)
rco2_p50 = np.percentile(rco2_tree_preds, 50)
rco2_p95 = np.percentile(rco2_tree_preds, 95)
# ==================================
# TARGET PROBABILITIES
# ==================================

prod_prob = (
    np.sum(prod_tree_preds > prod_target)
    / len(prod_tree_preds)
) * 100

rco2_prob = (
    np.sum(rco2_tree_preds > rco2_target)
    / len(rco2_tree_preds)
) * 100
# ==================================
# OUTPUT
# ==================================

print("\n")
print("=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

print(f"\nSpecies      : {species}")
print(f"CO2 (%)      : {co2}")
print(f"NaHCO3 (g/L) : {nahco3}")

print("\n")
print("-" * 60)
print(f"{'Metric':<20}{'Productivity':<18}{'RCO2'}")
print("-" * 60)

print(f"{'Predicted Mean':<20}{prod_mean:.4f}{'':<10}{rco2_mean:.4f}")
print(f"{'SD':<20}{prod_sd:.4f}{'':<10}{rco2_sd:.4f}")
print(f"{'P5':<20}{prod_p5:.4f}{'':<10}{rco2_p5:.4f}")
print(f"{'P50':<20}{prod_p50:.4f}{'':<10}{rco2_p50:.4f}")
print(f"{'P95':<20}{prod_p95:.4f}{'':<10}{rco2_p95:.4f}")

print("-" * 60)
print("LOOCV Productivity R² = 0.8241")
print("LOOCV RCO2 R²         = 0.8187")
print("-" * 60)
print("\nTARGET ANALYSIS")
print("-" * 60)

print(
    f"Probability(Productivity > {prod_target}) = "
    f"{prod_prob:.1f}%"
)

print(
    f"Probability(RCO2 > {rco2_target}) = "
    f"{rco2_prob:.1f}%"
)