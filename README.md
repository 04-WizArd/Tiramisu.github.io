
# ICI SE TROUVE LE MEMOIR EXPLIQUANT CHAQUE ETAPES DE LA REOLUTION DES QUESTION DES TRAVEAUX PRATIQUE
# Q1 : prediction Iris





## Import des bibliotheques et chargement du jeu des données Iris



```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

url = "ML/UCI.csv"
data = pd.read_csv(url, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

```

## Resumé du dataset et visualisation des données

```bash
print(data.shape)
print(data.head())
print(data.describe())

# Répartition des données par rapport à la variable de classe
print(data["species"].value_counts())

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
```

## Évaluation des algorithmes et estimation de la précision : Créer un jeu de test 20% et Mettre en place la validation croisée 10 folds

```bash
# Diviser les données en ensembles d'entraînement et de test (80%/20%)
X = data.drop("species", axis=1)
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    print(f"{model.__class__.__name__}: {scores.mean():.2f} ± {scores.std():.2f}"
```  

## Prédictions avec les 4 modèles sélectionnés sur le jeu de données Iris

```bash
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Initialiser les modèles
lr_model = LogisticRegression()
knn_model = KNeighborsClassifier(n_neighbors=5)
cart_model = DecisionTreeClassifier()
svm_model = SVC()

# Entraîner les modèles sur l'ensemble d'entraînement
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
cart_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Prédictions du modèle de régression logistique
lr_pred = lr_model.predict(X_test)

# Prédictions du modèle KNN
knn_pred = knn_model.predict(X_test)

# Prédictions du modèle CART
cart_pred = cart_model.predict(X_test)

# Prédictions du modèle SVM
svm_pred = svm_model.predict(X_test)

from sklearn.metrics import accuracy_score

# Évaluer la précision du modèle de régression logistique
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Précision du modèle de régression logistique: {lr_accuracy:.2f}")

# Évaluer la précision du modèle KNN
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"Précision du modèle KNN: {knn_accuracy:.2f}")

# Évaluer la précision du modèle CART
cart_accuracy = accuracy_score(y_test, cart_pred)
print(f"Précision du modèle CART: {cart_accuracy:.2f}")

# Évaluer la précision du modèle SVM
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"Précision du modèle SVM: {svm_accuracy:.2f}")

Sélectionner le meilleur modèle:
Sur la base des résultats de la validation croisée, le modèle SVC semble avoir la meilleure performance moyenne avec le moins d'écart-type.

``` 

## Faire des prédictions et évaluer la précision

```bash
model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur l'ensemble de test: {accuracy:.2f}")

confusion_matrix(y_test, y_pred)


