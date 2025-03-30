import numpy as np
import joblib

# Charger le modèle et le scaler
model = joblib.load('models/linear_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Préparer la nouvelle donnée
superficie =61
X_new = np.array([[superficie]])  # Notez les doubles crochets pour la forme 2D

# Normaliser la donnée (comme pendant l'entraînement)
X_new_scaled = scaler.transform(X_new)

# Faire la prédiction
prix_pred = model.predict(X_new_scaled)

# Afficher le résultat
print(f"Pour une maison de {superficie}m², le prix estimé est de {prix_pred[0]:.2f}€")