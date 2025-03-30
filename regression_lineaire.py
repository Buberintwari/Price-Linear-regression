# regression_lineaire.py

# === 1. Look at the big picture ===
"""
Objectif: Prédire le prix d'une maison en fonction de sa superficie.
Ce modèle pourra être utilisé par une agence immobilière pour estimer rapidement 
la valeur de propriétés.
"""

# Importations nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === 2. Get the data ===
def load_housing_data():
    # Création d'un dataset simple (en pratique, vous chargeriez un fichier CSV)
    data = {
        'superficie': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        'prix': [150000, 180000, 210000, 240000, 270000, 300000, 330000, 360000, 390000, 420000]
    }
    return pd.DataFrame(data)

housing = load_housing_data()

# === 3. Discover and visualize the data to gain insights ===
def explore_data(df):
    print("\n=== Exploration des données ===")
    print(df.head())
    print("\nDescription statistique:")
    print(df.describe())
    
    # Visualisation
    df.plot(kind='scatter', x='superficie', y='prix', alpha=0.5)
    plt.title('Prix des maisons en fonction de la superficie')
    plt.xlabel('Superficie (m²)')
    plt.ylabel('Prix (€)')
    plt.grid()
    plt.savefig('data_visualization.png')
    plt.close()

explore_data(housing)

# === 4. Prepare the data for Machine Learning algorithms ===
def prepare_data(df):
    # Séparation des features et de la target
    X = df[['superficie']].values
    y = df['prix'].values
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Séparation en train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = prepare_data(housing)

# === 5. Select a model and train it ===
def train_model(X_train, y_train):
    print("\n=== Entraînement du modèle ===")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Affichage des coefficients
print(f"\nCoefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# === 6. Fine-tune your model ===
def evaluate_model(model, X_test, y_test):
    print("\n=== Évaluation du modèle ===")
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Visualisation des prédictions
    plt.scatter(X_test, y_test, color='blue', label='Valeurs réelles')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prédictions')
    plt.title('Prédictions vs Valeurs réelles')
    plt.xlabel('Superficie normalisée')
    plt.ylabel('Prix')
    plt.legend()
    plt.savefig('predictions_vs_reality.png')
    plt.close()

evaluate_model(model, X_test, y_test)

# === 7. Present your solution ===
def save_model(model, scaler):
    # Création d'un dossier models s'il n'existe pas
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Sauvegarde du modèle et du scaler
    joblib.dump(model, 'models/linear_regression_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModèle et scaler sauvegardés dans le dossier 'models'")

save_model(model, scaler)

def make_prediction(model, scaler, superficie):
    # Préparation de la donnée d'entrée
    X_new = np.array([[superficie]])
    X_new_scaled = scaler.transform(X_new)
    
    # Prédiction
    prix_pred = model.predict(X_new_scaled)
    print(f"\nPour une superficie de {superficie}m², le prix estimé est de {prix_pred[0]:.2f}€")

# Exemple de prédiction
make_prediction(model, scaler, 75)  # Devrait donner environ 225000€

# === 8. Launch, monitor, and maintain your system ===
"""
Dans un environnement de production, vous pourriez:
1. Créer une API Flask/FastAPI pour servir le modèle
2. Mettre en place un monitoring des prédictions
3. Implémenter un système de re-entraînement périodique
4. Ajouter des logs pour le suivi
5. Mettre en place des tests unitaires

Exemple simplifié de monitoring:
"""

def log_prediction(features, prediction):
    with open('prediction_logs.csv', 'a') as f:
        f.write(f"{pd.Timestamp.now()},{features[0]},{prediction[0]}\n")

# Exemple d'utilisation
log_prediction([75], model.predict(scaler.transform([[75]])))

print("\n=== Processus terminé avec succès ===")