import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split

# TODO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandaritzar les dades: StandardScaler

# TODO
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train) # fit_transform
X_test_transformed = scaler.transform(X_test) # Usamos los valores del fit del train para transformar el test

# Entrenam una SVM linear (classe SVC)

# TODO
svc = SVC(C = 100, kernel="linear")
svc.fit(X_train_transformed, y_train)

# Prediccio
# TODO
y_prediction_svc = svc.predict(X_test_transformed)

# Metrica
# TODO
# Calculate by hand the accuracy
accuracy = np.sum(y_prediction_svc == y_test) / len(y_test)

# Compare with the accuracy of the SVC class
print("SVC accuracy: ", svc.score(X_test_transformed, y_test))
print("My accuracy: ", accuracy)