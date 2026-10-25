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
# SPECIES-SPECIFIC MEDIAN CV
# ==================================

if species == "CS":

    prod_cv = 3.95 / 100

    rco2_cv = 3.95 / 100

elif species == "SO":

    prod_cv = 2.86 / 100

    rco2_cv = 4.18 / 100

else:

    print("Invalid Species")

    exit()

# ==================================
# SD ESTIMATION
# ==================================

prod_sd = pred_prod * prod_cv

rco2_sd = pred_rco2 * rco2_cv

# ==================================
# MONTE CARLO
# ==================================

n = 10000

prod_sim = np.random.normal(
    pred_prod,
    prod_sd,
    n
)

rco2_sim = np.random.normal(
    pred_rco2,
    rco2_sd,
    n
)

# ==================================
# PRODUCTIVITY STATS
# ==================================

prod_mean = np.mean(prod_sim)

prod_p5 = np.percentile(prod_sim, 5)

prod_p50 = np.percentile(prod_sim, 50)

prod_p95 = np.percentile(prod_sim, 95)

prod_prob = (
    np.sum(prod_sim > prod_target)
    / len(prod_sim)
) * 100

# ==================================
# RCO2 STATS
# ==================================

rco2_mean = np.mean(rco2_sim)

rco2_p5 = np.percentile(rco2_sim, 5)

rco2_p50 = np.percentile(rco2_sim, 50)

rco2_p95 = np.percentile(rco2_sim, 95)

rco2_prob = (
    np.sum(rco2_sim > rco2_target)
    / len(rco2_sim)
) * 100

# ==================================
# OUTPUT
# ==================================

print("\n")
print("=" * 65)
print("FINAL HYBRID MODEL")
print("=" * 65)

print(f"\nSpecies      : {species}")
print(f"CO2 (%)      : {co2}")
print(f"NaHCO3 (g/L) : {nahco3}")

print("\nPRODUCTIVITY")
print("-" * 65)

print(f"Predicted Mean : {prod_mean:.4f}")
print(f"SD             : {prod_sd:.4f}")
print(f"P5             : {prod_p5:.4f}")
print(f"P50            : {prod_p50:.4f}")
print(f"P95            : {prod_p95:.4f}")

print(
    f"Probability(Productivity > {prod_target}) = "
    f"{prod_prob:.2f}%"
)

print("\nRCO2")
print("-" * 65)

print(f"Predicted Mean : {rco2_mean:.4f}")
print(f"SD             : {rco2_sd:.4f}")
print(f"P5             : {rco2_p5:.4f}")
print(f"P50            : {rco2_p50:.4f}")
print(f"P95            : {rco2_p95:.4f}")

print(
    f"Probability(RCO2 > {rco2_target}) = "
    f"{rco2_prob:.2f}%"
)

print("=" * 65)