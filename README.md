		                                      ICI SE TROUVE LE MEMOIR EXPLIQUANT CHAQUE ETAPES DE LA REOLUTION DES QUESTION DES TRAVEAUX PRATIQUE
		                                      ===================================================================================================


Projet Machine Learning: Classification des Iris
1. Chargement des données
Importer les bibliothèques:

Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


Charger le jeu de données Iris:

Python

# Télécharger le jeu de données Iris depuis le référentiel UCI Machine Learning
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])


Résumer le jeu de données:

Python

# Afficher les dimensions du jeu de données
print(data.shape)

# Aperçu des données
print(data.head())

# Résumé statistique
print(data.describe())

# Répartition des données par rapport à la variable de classe
print(data["species"].value_counts())


Visualiser les données:

Python

# Histogrammes pour chaque caractéristique
data.hist(figsize=(10, 5))
plt.tight_layout()
plt.show()

# Diagrammes à points pour chaque paire de caractéristiques
for i in range(4):
    for j in range(i + 1, 4):
        plt.scatter(data["sepal_length"], data["petal_length"], c=data["species"])
        plt.title(f"Sepal Length vs Petal Length - {data['species'][i]} & {data['species'][j]}")
        plt.show()


2. Évaluation des algorithmes et estimation de la précision
Créer un jeu de test (20%) :

Python

# Diviser les données en ensembles d'entraînement et de test (80%/20%)
X = data.drop("species", axis=1)
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Mettre en place la validation croisée 10 folds:

Python

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


Sélectionner le meilleur modèle:
Sur la base des résultats de la validation croisée, le modèle SVC semble avoir la meilleure performance moyenne avec le moins d'écart-type.
3. Faire des prédictions et évaluer la précision
Entraîner le modèle SVC sur l'ensemble d'entraînement:

Python

model = SVC()
model.fit(X_train, y_train)


Faire des prédictions sur l'ensemble de test:

Python

y_pred = model.predict(X_test)


Évaluer la précision sur l'ensemble de test:

Python

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur l'ensemble de test: {accuracy:.2f}")


Analyser la matrice de confusion:

Python

confusion_matrix(y_test, y_pred)
