# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
import os
from os import path
lokalizacja=os.getcwd()
lokalizacja

# Sprawdzam czy plik z oryginalnymi danymi znajduje się w  tej samej lokalizacji co plik z kodem
if path.isfile("processed.cleveland.data")==True:
        print("Plik znajduje sie w tym samej lokalizacji co plik z kodem")
else:
        print("Plik z danymi nie znajduje sie w tym samej lokalizacji co plik z kodem")

dane_odczytu ="\processed.cleveland.data"
dane_zapisu ="\heart.csv"
lokalizacja=os.getcwd()
sciezka_odczytu=lokalizacja+dane_odczytu
sciezka_zapisu=lokalizacja+dane_zapisu 
print("Scieżka odczytu pliku z lokalizacji:")
print(sciezka_odczytu)
print("Scieżka zapisu:")
print(sciezka_zapisu)


#The authors of the databases have requested that any publications resulting from the use of the data include the names of the principal investigator responsible for the data collection at each institution. They would be:
#1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
#2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
#3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
#4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D. r'


dane_oryginalne = pd.read_csv (open(sciezka_odczytu,'r'),header=None
                         , names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak'
                                  ,'slope','ca','thal','num'])
dane_oryginalne.to_csv (open(sciezka_zapisu,'w'), index=None)



# oryginalne dane
with open(sciezka_odczytu,'r') as plik:
    zawartosc=plik.read()
    print(zawartosc)

# zapisane dane
with open(sciezka_zapisu,'r') as plik:
    zawartosc=plik.read()
    print(zawartosc)

dane_oryginalne.head()

plik_all = "heart.csv"
heart_cleveland = pd.read_csv(plik_all , sep = ",")

heart_cleveland.head()

#kształt danych
print("Kształt danych:")
print(heart_cleveland .shape) 

#w danych są znaki specjalne - wczytujemy te dane jako braki danych
znaki_specjalne_i_braki_danych= ['~','!','@','#','$','%','^','&','*','(',')','-','=',
                                 '{','}','|',':','"','<','>','/','*','-','+','~_','+',
                                 '{','}','|',':','"','<','>','?']

#wczytujemy zbiór danych z zamianą znaków specjalnych na NA
heart_data_brudne= pd.read_csv("heart.csv", na_values =znaki_specjalne_i_braki_danych)

#Sprawdzmy ile mamy braków danych
heart_datacos= pd.DataFrame(heart_data_brudne)
heart_datacos.isna().sum().sum()
print("Liczba rekordów brakami danych:")
print(heart_datacos.isna().sum().sum())
#Mamy 6 wierszy z brakami danych, prawdopodobnie usuniemy te wiersze z analizy
#Możemy je także zastąpic wartościami średnymi, modalnymi lub losowymi 

heart_data_brudne.head()

#dodajemy kolumne output: 0 - bez choroby, 1 - z chorobą serca (wartosci num od 1-4)
heart_data_brudne['output']=np.where(heart_data_brudne['num']<1,0,1)

#dane po dodaniu kolumny output
heart_data_brudne.head()

#usuwamy zbedną kolumne num
heart_data_brudne=heart_data_brudne.drop(columns=['num'])

heart_data_brudne.head()

#kształt danych
print("Kształt danych:")
print(heart_data_brudne.shape) 

#wykaz wszystkich kolumn wraz z typem kazdej zmiennej
print("Ogólna informacja o danych:")
print(heart_data_brudne.info())

#większość zmiennych jest typu float64
#część z nich zostanie później zamieniona na category
#iterując po poszczególnych kolumnach, wyświetlę jeszcze liczbę pustych obserwacji dla każdej zmiennej
for column in heart_data_brudne:
    print(column, "- Puste obserwacje: ", heart_data_brudne[column].isnull().sum())

#mamy 6 pustych obeserwacji, które wcześniej były znakami zapytania
#zobaczmy jak wyglądają rekordy z pustymi ca
heart_data_brudne[heart_data_brudne.ca.isnull()]

#zobaczmy jak wyglądają rekordy z pustymi thal
heart_data_brudne[heart_data_brudne.thal.isnull()]

#usuwamy wiersze z pustymi obserwacjami
heart_data = heart_data_brudne.dropna()

#podział na zbiór testowy i uczący
y = heart_data['output']
X = heart_data.drop('output', 1)

import sklearn.model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=12345)
#print(X_ucz.shape)
print(X_test.shape)
#print(y_ucz.shape)
print(y_test.shape)

#sprawdzanie parametrów algorytmu
def fit_classifier(alg, X_train, X_test, y_train, y_test):
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    #y_pred = np.argmax(y_pred, axis=0)
    return {
        "ACC": sklearn.metrics.accuracy_score(y_pred, y_test),
        "P":   sklearn.metrics.precision_score(y_pred, y_test, average='weighted'),
        "R":   sklearn.metrics.recall_score(y_pred, y_test, average='weighted'),
        "F1":  sklearn.metrics.f1_score(y_pred, y_test, average='weighted')
    }

### drzewo
import sklearn.tree
drzewo = sklearn.tree.DecisionTreeClassifier()
drzewo.fit(X_train, y_train)

feature = drzewo.feature_importances_
pd.Series(drzewo.feature_importances_, index = X_train.columns[0:13]).sort_values(ascending=False)

# predykcja na zbiorze testowym
y_pred_test = drzewo.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = drzewo.predict(X_train)

# sprawdzenie accuracy score
print('Random tree model accuracy test set score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#test set score : 0.6833
print('Random tree model accuracy training set score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#trainig set score : 1.000

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_drzewo = cross_val_score(drzewo, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_drzewo), std(n_scores_drzewo)))
#Średni accuracy score po kroswalidacji: 0.727, odchylenie standardowe: 0.089

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(drzewo, X_train, X_test, y_train, y_test))

### las losowy
import sklearn.ensemble
las = sklearn.ensemble.RandomForestClassifier(random_state=123)
las.fit(X_train, y_train)

#poziom istotności zmiennych w lesie losowym
pd.Series(las.feature_importances_, index = X.columns[0:13]).sort_values(ascending=False)

# predykcja na zbiorze testowym
y_pred_test = las.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = las.predict(X_train)

# sprawdzenie accuracy score
from sklearn.metrics import accuracy_score
print('Random forest model accuracy test set score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#test set score : 0.75
print('Random forest model accuracy training set score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#trainig set score : 1.000

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_las = cross_val_score(las, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_las), std(n_scores_las)))
#Średni accuracy score po kroswalidacji: 0.813, odchylenie standardowe: 0.067

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(las, X_train, X_test, y_train, y_test))

#BOOSTING

### XGBoost
pip install xgboost
import xgboost as xgb

# zdefiniowanie data_dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y)

# podział na zbiór testowy i treningowy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# import XGBClassifier
from xgboost import XGBClassifier

# deklareacja parametrów
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }

# instancja klasyfikatora
xgb_clf = XGBClassifier(**params)

# dopasowanie klasyfikatora do zbioru treningowego
xgb_clf.fit(X_train, y_train)

# predykcja na zbiorze testowym
y_pred_test = xgb_clf.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = xgb_clf.predict(X_train)


# sprawdzenie accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy test score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#score test: 0.7333
print('XGBoost model accuracy train score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#score trainig: 0.8987

# kroswalidacja
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_xgb = cross_val_score(xgb_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_xgb), std(n_scores_xgb)))
#Średni accuracy score po kroswalidacji: 0.798, odchylenie standardowe: 0.081

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(xgb_clf, X_train, X_test, y_train, y_test))

#wyniki
xgb.plot_importance(xgb_clf)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

# tuning - drugi model
params2 = {
            'objective':'binary:logistic',
            'max_depth': 7,
            'alpha': 10,
            'learning_rate': 0.5,
            'n_estimators':100
        }
                    
# instancja klasyfikatora
xgb_clf2 = XGBClassifier(**params2)

# dopasowanie klasyfikatora do zbioru treningowego
xgb_clf2.fit(X_train, y_train)

# predykcja na zbiorze testowym
y_pred_test = xgb_clf2.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = xgb_clf2.predict(X_train)

# sprawdzenie accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy test score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#score test: 0.7333
print('XGBoost model accuracy train score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#score trainig: 0.9072

# kroswalidacja
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_xgb = cross_val_score(xgb_clf2, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_xgb), std(n_scores_xgb)))
#Średni accuracy score po kroswalidacji: 0.807, odchylenie standardowe: 0.081

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(xgb_clf2, X_train, X_test, y_train, y_test))

xgb.plot_importance(xgb_clf2)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

# tuning - trzeci model - NAJLEPSZY
params3 = {
            'objective':'binary:logistic',
            'max_depth': 9,
            'alpha': 10,
            'learning_rate': 0.1,
            'n_estimators':100
        }
                            
# instancja klasyfikatora
xgb_clf3 = XGBClassifier(**params3)

# dopasowanie klasyfikatora do zbioru treningowego
xgb_clf3.fit(X_train, y_train)

# predykcja na zbiorze testowym
y_pred_test = xgb_clf3.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = xgb_clf3.predict(X_train)

# sprawdzenie accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy test score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#score test: 0.75
print('XGBoost model accuracy train score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#score trainig: 0.9072

# kroswalidacja
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_xgb = cross_val_score(xgb_clf3, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_xgb), std(n_scores_xgb)))
#Średni accuracy score po kroswalidacji: 0.812, odchylenie standardowe: 0.089

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(xgb_clf3, X_train, X_test, y_train, y_test))

xgb.plot_importance(xgb_clf3)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

# tuning - czwarty model 

params4 = {
            'objective':'binary:logistic',
            'max_depth': 9,
            'alpha': 10,
            'learning_rate': 0.05,
            'n_estimators':100
        }
          
                    
# instancja klasyfikatora
xgb_clf4 = XGBClassifier(**params4)

# dopasowanie klasyfikatora do zbioru treningowego
xgb_clf4.fit(X_train, y_train)

# predykcja na zbiorze testowym
y_pred_test = xgb_clf4.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = xgb_clf4.predict(X_train)

# sprawdzenie accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy test score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#score test: 0.7667
print('XGBoost model accuracy train score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#score trainig: 0.9114

# kroswalidacja
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_xgb = cross_val_score(xgb_clf4, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_xgb), std(n_scores_xgb)))
#Średni accuracy score po kroswalidacji: 0.810, odchylenie standardowe: 0.089

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(xgb_clf4, X_train, X_test, y_train, y_test))

xgb.plot_importance(xgb_clf4)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

# tuning - piąty model

params5 = {
            'objective':'binary:logistic',
            'max_depth': 9,
            'alpha': 10,
            'learning_rate': 0.15,
            'n_estimators':100
        }      
                   
# instancja klasyfikatora
xgb_clf5 = XGBClassifier(**params5)

# dopasowanie klasyfikatora do zbioru treningowego
xgb_clf5.fit(X_train, y_train)

# predykcja na zbiorze testowym
y_pred_test = xgb_clf5.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = xgb_clf5.predict(X_train)

# sprawdzenie accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy test score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#score test: 0.7167
print('XGBoost model accuracy train score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#score trainig: 0.8987

# kroswalidacja
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_xgb = cross_val_score(xgb_clf5, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_xgb), std(n_scores_xgb)))
#Średni accuracy score po kroswalidacji: 0.810, odchylenie standardowe: 0.089

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(xgb_clf5, X_train, X_test, y_train, y_test))

xgb.plot_importance(xgb_clf5)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()


### Light GBM
pip install lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier

y = heart_data['output']
X = heart_data.drop('output', 1)

import sklearn.model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=12345)

#zbudowanie modelu
lgbm_clf = lgb.LGBMClassifier()
lgbm_clf.fit(X_train, y_train)

# predykcja na zbiorze testowym
y_pred_test = lgbm_clf.predict(X_test)
# predykcja na zbiorze uczącym
y_pred_train = lgbm_clf.predict(X_train)

# sprawdzenie accuracy score
print('Light GBM model accuracy test score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#score test: 0.7667
print('Light GBM model accuracy train score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#score trainig: 1.000

#kroswalidacja
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores_lgbm = cross_val_score(lgbm_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores_lgbm), std(n_scores_lgbm)))
#Średni accuracy score po kroswalidacji: 0.798, odchylenie standardowe: 0.083

#korzystamy z funkcji fit_classifier
pd.Series(fit_classifier(lgbm_clf, X_train, X_test, y_train, y_test))

lgb.plot_importance(lgbm_clf)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()
