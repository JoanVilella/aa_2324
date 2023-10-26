from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures


X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler() #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# SVM
svc = SVC(kernel='linear', C=1.0)
svc.fit(X_transformed, y_train)
y_pred = svc.predict(X_test_transformed)
print("SVM: ", precision_score(y_test, y_pred))

# SVC with kernel_lineal
def kernel_lineal(x1, x2):
     return x1.dot(x2.T)

my_svc = SVC(kernel=kernel_lineal, C=1.0)
my_svc.fit(X_transformed, y_train)
y_pred = my_svc.predict(X_test_transformed)
print("SVC with kernel_lineal: ", precision_score(y_test, y_pred))

# SVC with RBF
svc_rbf = SVC(kernel='rbf', C=1.0, gamma=10)
svc_rbf.fit(X_transformed, y_train)
y_pred = svc_rbf.predict(X_test_transformed)
print("SVC with RBF: ", precision_score(y_test, y_pred))

# SVC with kernek_RBF
def kernel_RBF(x1, x2, gamma = 10):
    dm = distance_matrix(x1, x2)
    return np.exp(-gamma * dm ** 2)

svc_my_rbf = SVC(kernel=kernel_RBF, C=1.0)
svc_my_rbf.fit(X_transformed, y_train)
y_pred = svc_my_rbf.predict(X_test_transformed)
print("SVC with kernel_RBF: ", precision_score(y_test, y_pred))


# SVC with polynomial kernel with scikit
svc_poly = SVC(kernel='poly', C=1.0, degree=3)
svc_poly.fit(X_transformed, y_train)
y_pred = svc_poly.predict(X_test_transformed)
print("SVC with polynomial kernel with scikit: ", precision_score(y_test, y_pred))

# SVC with polynomial kernel
def kernel_poly(x1, x2, degree=3, gamma=10):
    return (gamma * x1.dot(x2.T)) ** degree # Gamma porque Yolo

svc_my_poly = SVC(kernel=kernel_poly, C=1.0)
svc_my_poly.fit(X_transformed, y_train)
y_pred = svc_my_poly.predict(X_test_transformed)
print("SVC with polynomial kernel: ", precision_score(y_test, y_pred))

# Feina 4
# PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_transformed)
X_test_poly = poly.transform(X_test_transformed)

# Use a linear SVC
svc_linear_poly = SVC(kernel='linear', C=1.0)
svc_linear_poly.fit(X_poly, y_train)
y_pred = svc_linear_poly.predict(X_test_poly)
print("SVC with polynomial features: ", precision_score(y_test, y_pred))







