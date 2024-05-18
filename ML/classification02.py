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


# Télécharger le jeu de données depuis l'URL fournie
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151) 
  
# data (as pandas dataframes) 
X = connectionist_bench_sonar_mines_vs_rocks.data.features 
y = connectionist_bench_sonar_mines_vs_rocks.data.targets 
  
# metadata 
print(connectionist_bench_sonar_mines_vs_rocks.metadata) 
  
# variable information 
print(connectionist_bench_sonar_mines_vs_rocks.variables) 


data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/sonar/sonar.data"
data = pd.read_csv(data_url, header=None, names=["Rock or Mine", *range(60)])


# Afficher la forme et le type des données
print(data.shape)
print(data.dtypes)

# Aperçu des données
print(data.head())

