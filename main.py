# =============================================================================
#  PRÉVISION DES VENTES PAR MARQUE ET VILLE — Motors.tn
#  Comparaison : Régression de Poisson | Random Forest | XGBoost
#  Agrégation annuelle (Ville × Marque × Année)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Chargement des données brutes
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 2 — Chargement des données brutes")
print("=" * 60)

df = pd.read_csv("Répartition_par_ville_-_Motors_tn (7).csv")

print(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
print("\nAperçu :")
print(df.head())
print("\nValeurs manquantes :")
print(df.isnull().sum())

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Nettoyage & préparation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Nettoyage & préparation")
print("=" * 60)

df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
df['Année'] = df['Date'].dt.year

print(f"Période : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Années présentes : {sorted(df['Année'].unique())}")

# Suppression des colonnes non pertinentes ou à variance nulle
# CARROSSERIE, ENERGIE, Essieus, Places assises, TYPE = valeur unique → aucun apport
colonnes_a_supprimer = [
    'Modèle', 'Essieus', 'Places assises', 'CARROSSERIE',
    'ENERGIE', 'TYPE', 'Numéro ligne', 'USAGE', 'QUALITE'
]
df = df.drop(columns=colonnes_a_supprimer)

print(f"\nColonnes retenues : {df.columns.tolist()}")
print(f"Valeurs manquantes après nettoyage :\n{df.isnull().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — Agrégation annuelle : Ville × Marque × Année
#
# Pourquoi annuel et non mensuel ?
# Le dataset contient 907 enregistrements sur 5 ans répartis sur
# 26 villes × 18 marques. Une agrégation mensuelle produit des groupes
# quasi-vides (88 % à 1 vente) ce qui empêche tout apprentissage.
# L'agrégation annuelle donne une distribution bien plus variée
# (std = 1.35, max = 9) avec laquelle les modèles peuvent apprendre.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — Feature Engineering & Agrégation annuelle")
print("=" * 60)

df_agg = df.groupby(
    ['Ville', 'Marque', 'Année']
).agg(
    Ventes=('Marque', 'count'),
    PUISSANCE_MOY=('PUISSANCE', 'mean'),
    PTAC_MOY=('PTAC', 'mean')
).reset_index()

print(f"Dataset agrégé : {df_agg.shape[0]} lignes | {df_agg.shape[1]} colonnes")
print("\nDistribution de la cible Ventes :")
print(df_agg['Ventes'].describe())
print("\nFréquences :")
print(df_agg['Ventes'].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────────────────────
# Encodage des variables catégorielles
# Un LabelEncoder distinct par colonne → stocké dans un dictionnaire
# pour pouvoir décoder ou réutiliser lors des prédictions futures
# ─────────────────────────────────────────────────────────────────────────────
colonnes_categorielles = ['Ville', 'Marque']

encoders = {}
for col in colonnes_categorielles:
    le = LabelEncoder()
    df_agg[col + '_enc'] = le.fit_transform(df_agg[col])
    encoders[col] = le
    print(f"{col} — {df_agg[col].nunique()} modalités encodées")

joblib.dump(encoders, "encoders.pkl")
print("\nEncodeurs sauvegardés dans encoders.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — Définition des features & Split Train / Test
#
# Split temporel :
#   Train : 2021-2023  (données historiques complètes)
#   Test  : 2024-2025  (années récentes — 2026 exclue car incomplète)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Définition des features & Split Train / Test")
print("=" * 60)

colonnes_a_exclure = ['Ville', 'Marque', 'PAYS', 'Ventes']

features = [col for col in df_agg.columns if col not in colonnes_a_exclure]
print(f"Features utilisées ({len(features)}) : {features}")

X = df_agg[features]
y = df_agg['Ventes']

train_mask = df_agg['Année'] <= 2023
test_mask  = df_agg['Année'].isin([2024, 2025])

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain (2021-2023) : {X_train.shape}")
print(f"Test  (2024-2025) : {X_test.shape}")
print(f"\nDistribution Ventes TRAIN :\n{y_train.value_counts().sort_index()}")
print(f"\nDistribution Ventes TEST :\n{y_test.value_counts().sort_index()}")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — Modèle 1 : Régression de Poisson
#
# Justification : la cible est un comptage d'engins immatriculés (entiers ≥ 1).
# La régression de Poisson est le modèle de référence pour les données de
# comptage et sert ici de baseline statistique.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Modèle 1 : Régression de Poisson")
print("=" * 60)

poisson = TweedieRegressor(power=1, alpha=0, max_iter=1000)
poisson.fit(X_train, y_train)
y_pred_poisson = poisson.predict(X_test)

mae_poisson  = mean_absolute_error(y_test, y_pred_poisson)
rmse_poisson = np.sqrt(mean_squared_error(y_test, y_pred_poisson))
r2_poisson   = r2_score(y_test, y_pred_poisson)

print(f"MAE  : {mae_poisson:.4f}")
print(f"RMSE : {rmse_poisson:.4f}")
print(f"R²   : {r2_poisson:.4f}")

coefficients = pd.DataFrame({
    'Feature'     : X_train.columns,
    'Coefficient' : poisson.coef_
}).sort_values('Coefficient', ascending=False)
print("\nCoefficients (importance des features) :")
print(coefficients.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 — Modèle 2 : Random Forest
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 7 — Modèle 2 : Random Forest")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

print(f"MAE  : {mae_rf:.4f}")
print(f"RMSE : {rmse_rf:.4f}")
print(f"R²   : {r2_rf:.4f}")

importance_rf = pd.DataFrame({
    'Feature'   : X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nImportance des variables :")
print(importance_rf.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 8 — Modèle 3 : XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 8 — Modèle 3 : XGBoost")
print("=" * 60)

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb   = r2_score(y_test, y_pred_xgb)

print(f"MAE  : {mae_xgb:.4f}")
print(f"RMSE : {rmse_xgb:.4f}")
print(f"R²   : {r2_xgb:.4f}")

importance_xgb = pd.DataFrame({
    'Feature'   : X_train.columns,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nImportance des variables :")
print(importance_xgb.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 9 — Comparaison des modèles
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 9 — Comparaison des modèles")
print("=" * 60)

resultats_modeles = pd.DataFrame({
    'Modèle': ['Poisson', 'Random Forest', 'XGBoost'],
    'MAE'   : [mae_poisson, mae_rf, mae_xgb],
    'RMSE'  : [rmse_poisson, rmse_rf, rmse_xgb],
    'R²'    : [r2_poisson, r2_rf, r2_xgb]
})

print(resultats_modeles.to_string(index=False))
print("\nClassement RMSE (↓ meilleur) :")
print(resultats_modeles.sort_values('RMSE').to_string(index=False))
print("\nClassement R² (↑ meilleur) :")
print(resultats_modeles.sort_values('R²', ascending=False).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 10 — Sélection automatique du meilleur modèle (critère : RMSE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 10 — Sélection automatique du meilleur modèle")
print("=" * 60)

modeles_dict = {
    'Poisson'      : (poisson, y_pred_poisson),
    'Random Forest': (rf,      y_pred_rf),
    'XGBoost'      : (xgb,     y_pred_xgb)
}

meilleur_idx               = resultats_modeles['RMSE'].idxmin()
meilleur_nom               = resultats_modeles.loc[meilleur_idx, 'Modèle']
modele_final, y_pred_final = modeles_dict[meilleur_nom]

print(f"✅ Meilleur modèle : {meilleur_nom}")
print(f"   MAE  = {resultats_modeles.loc[meilleur_idx, 'MAE']:.4f}")
print(f"   RMSE = {resultats_modeles.loc[meilleur_idx, 'RMSE']:.4f}")
print(f"   R²   = {resultats_modeles.loc[meilleur_idx, 'R²']:.4f}")

joblib.dump(modele_final, "model_final.pkl")
print(f"\nModèle sauvegardé dans model_final.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 11 — Aperçu comparatif des prédictions sur le jeu de test
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 11 — Aperçu comparatif des prédictions")
print("=" * 60)

comparaison_finale = pd.DataFrame({
    'Ville'        : df_agg.loc[test_mask, 'Ville'].values,
    'Marque'       : df_agg.loc[test_mask, 'Marque'].values,
    'Année'        : df_agg.loc[test_mask, 'Année'].values,
    'Réel'         : y_test.values,
    'Poisson'      : np.round(y_pred_poisson, 2),
    'Random_Forest': np.round(y_pred_rf, 2),
    'XGBoost'      : np.round(y_pred_xgb, 2)
})
comparaison_finale['Erreur_Poisson'] = np.round(np.abs(comparaison_finale['Réel'] - comparaison_finale['Poisson']), 2)
comparaison_finale['Erreur_RF']      = np.round(np.abs(comparaison_finale['Réel'] - comparaison_finale['Random_Forest']), 2)
comparaison_finale['Erreur_XGB']     = np.round(np.abs(comparaison_finale['Réel'] - comparaison_finale['XGBoost']), 2)

print(comparaison_finale.head(20).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 12 — Construction du dataset futur 2026
#
# On prend les combinaisons uniques Ville × Marque observées dans les données
# historiques et on prédit pour l'année 2026 complète.
# (2026 est partiellement présente dans le dataset → on prédit l'année entière)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 12 — Prévisions pour 2026")
print("=" * 60)

colonnes_stables = ['Ville', 'Marque', 'Ville_enc', 'Marque_enc',
                    'PUISSANCE_MOY', 'PTAC_MOY']

df_unique = df_agg[colonnes_stables].drop_duplicates().reset_index(drop=True)
print(f"Combinaisons uniques Ville × Marque : {len(df_unique)}")

df_futur          = df_unique.copy()
df_futur['Année'] = 2026

X_futur               = df_futur[features]
pred_futur            = modele_final.predict(X_futur)
df_futur['Prévision'] = np.round(pred_futur, 2)
df_futur['Prévision_arrondie'] = np.round(pred_futur).astype(int)

print(f"\nAperçu des prévisions 2026 :")
print(df_futur[['Ville', 'Marque', 'Année', 'Prévision', 'Prévision_arrondie']].head(20).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 13 — Agrégations des prévisions 2026
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 13 — Agrégations des prévisions 2026")
print("=" * 60)

def agreger(df, groupby_cols, col='Prévision'):
    return (
        df.groupby(groupby_cols)
        .agg(
            Prévision_totale=(col, 'sum'),
            Prévision_moyenne=(col, 'mean'),
            Nombre_combinaisons=(col, 'count')
        )
        .reset_index()
        .round(2)
        .sort_values('Prévision_totale', ascending=False)
    )

prev_par_marque       = agreger(df_futur, 'Marque')
prev_par_ville        = agreger(df_futur, 'Ville')
prev_par_marque_ville = agreger(df_futur, ['Ville', 'Marque'])

print("TOP 10 marques prévues en 2026 :")
print(prev_par_marque.head(10).to_string(index=False))

print("\nTOP 10 villes prévues en 2026 :")
print(prev_par_ville.head(10).to_string(index=False))

print("\nTOP 10 combinaisons Ville + Marque prévues en 2026 :")
print(prev_par_marque_ville.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 14 — Résumé global des prévisions 2026
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 14 — Résumé global des prévisions 2026")
print("=" * 60)

print(f"Total général prévu    : {df_futur['Prévision'].sum():.2f}")
print(f"Moyenne des prévisions : {df_futur['Prévision'].mean():.2f}")
print(f"Minimum prévu          : {df_futur['Prévision'].min():.2f}")
print(f"Maximum prévu          : {df_futur['Prévision'].max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 15 — Export des résultats
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 15 — Export des résultats")
print("=" * 60)

colonnes_affichage = ['Ville', 'Marque', 'Année',
                      'PUISSANCE_MOY', 'PTAC_MOY',
                      'Prévision', 'Prévision_arrondie']

df_futur[colonnes_affichage].to_csv(
    "prevision_2026_detaillee.csv",       index=False, encoding='utf-8-sig')
prev_par_marque.to_csv(
    "prevision_2026_par_marque.csv",      index=False, encoding='utf-8-sig')
prev_par_ville.to_csv(
    "prevision_2026_par_ville.csv",       index=False, encoding='utf-8-sig')
prev_par_marque_ville.to_csv(
    "prevision_2026_par_marque_ville.csv",index=False, encoding='utf-8-sig')
comparaison_finale.to_csv(
    "comparaison_predictions_test.csv",   index=False, encoding='utf-8-sig')
resultats_modeles.to_csv(
    "comparaison_modeles.csv",            index=False, encoding='utf-8-sig')

print("Fichiers exportés :")
print("  - prevision_2026_detaillee.csv")
print("  - prevision_2026_par_marque.csv")
print("  - prevision_2026_par_ville.csv")
print("  - prevision_2026_par_marque_ville.csv")
print("  - comparaison_predictions_test.csv")
print("  - comparaison_modeles.csv")
print("  - model_final.pkl")
print("  - encoders.pkl")
print("\n✅ Pipeline complet terminé avec succès.")