""""


                                                TRAVAIL PRATIQUE DE MACHINE LEARNING 
                                                =====================================
                                                RESOLUTION QUESTION N°2 : Sonar Mines vs Rocks
                                                ===========================================

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


# Télécharger le jeu de données
data_url = "ML/sonar.csv"
data = pd.read_csv(data_url, header=None, names=["Rock or Mine", *range(60)])

# Aperçu des données
print(data.shape)
print(data.dtypes)

print(data.head())
print(data.describe())
print(data["Rock or Mine"].value_counts())

data.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
corr = data.corr()
plt.matshow(corr)
plt.show()

# Diviser les données en ensembles d'entraînement et de test (80%/20%)
X = data.drop("Rock or Mine", axis=1)
y = data["Rock or Mine"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#val croisee 10 fold
from sklearn.model_selection import cross_val_score

# Définir la fonction d'évaluation (précision)
def cv_accuracy(model, X, y):
    return cross_val_score(model, X, y, cv=10, scoring="accuracy")

# Évaluer les modèles avec la validation croisée
models = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    SVC(),
]

for model in models:
    scores = cv_accuracy(model, X_train, y_train)
    print(f"{model.__class__.__name__}: {scores.mean():.2f} ± {scores.std():.2f}")

"""     Sur la base des résultats de la validation croisée, le modèle SVC semble avoir
        la meilleure performance moyenne avec le moins d'écart-type."""


#Standardiser les données:
from sklearn.pipeline import Pipeline

scaler = StandardScaler()
model = SVC()

pipeline = Pipeline([("scaler", scaler), ("model", model)])

pipeline.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Évaluer la précision sur l'ensemble de test standardisé
print("Précision sur ensemble de test standardisé:", accuracy_score(y_test, y_pred))

#Réglage des algorithmes
# Ajuster les paramètres de SVC avec GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["auto", "scale"],
}

grid_search = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=10)
grid_search.fit(X_train, y_train)

print("Meilleurs paramètres SVC:", grid_search.best_params_)


# Ajuster le paramètre n_neighbors de KNeighborsClassifier avec GridSearchCV
param_grid = {"n_neighbors": range(3, 21)}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring="accuracy", cv=10)
grid_search.fit(X_train, y_train)

print("Meilleur paramètre n_neighbors:", grid_search.best_params_)
    
# Entraîner le modèle final (SVC ou RandomForestClassifier) sur l'ensemble de données complet
model.fit(X, y)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer la précision sur l'ensemble de test
print("Précision sur l'ensemble de test:", accuracy_score(y_test, y_pred))

# Entraîner le modèle SVC sur l'ensemble de données standardisé
X_scaled = scaler.transform(X)
y_pred_scaled = model.predict(X_scaled)

# Évaluer la précision sur l'ensemble de test standardisé
print("Précision sur ensemble de test standardisé:", accuracy_score(y_test, y_pred_scaled))






