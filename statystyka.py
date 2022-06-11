# -*- coding: utf-8 -*-
# In[1]:
# Dane pochodza 
#https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ 
 
#"Uwaga od  autorów ktory przeprowadzili oryginalne badanie, kazdorazowe użycie zbioru wymaga podanie ich 
# podania ich w danej publikacji":
#1. "Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
#2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
#3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
#4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D. r'""

###############################################################################

# Tytuł projektu: Klasyfikacja czynników wpływających na wystąpienie chorób serca
#
# Źródło danych: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
#
# Wykaz kolumn w pliku w raz z ich krótkim opisem:
#
# age : wiek pacjenta w latach
#
# sex : płeć pacjenta -> zmienna kategoryczna
#       wartość 1: mężczyzna 
#       wartość 0: kobieta
#
# cp : poziom cp pacjenta,  tj. rodzaj bólu w klatce piersiowej -> zmienna kategoryczna:
#      wartość 3 : ból nie związany z dusznicą bolesną
#      wartość 2 : nietypowa dusznica bolesna
#      wartość 1 : typowa dusznica bolesna
#      wartość 4 : bezobjawowy
#
# trestbps : spoczynkowe ciśnienie krwi (w mm Hg) przy przyjęciu do szpitala
#
# chol : poziom cholesterolu pacjenta w mg/dl
#
# fbs : poziom FBS pacjenta, tj. cukier we krwi na czczo > 120 mg/dl -> zmienna kategoryczna
#       wartość 1 : prawda
#       wartość 0 : fałsz
#
# restecg : spoczynkowe wyniki elektrokardiograficzne EKG pacjenta -> zmienna kategoryczna
#      wartość 0: prawidłowe
#      wartość 1: z nieprawidłowościami załamka ST-T (odwrócenie załamka T i/lub uniesienie lub obniżenie odcinka ST > 0,05 mV)
#      wartość 2: zgodnie ze standardem Estes pokazujące możliwy lub wyraźny przerost lewej komory - stan ciężki
#
# thalach : osiągnięte maksymalne tętno
#
# exang : dławica piersiowa (dusznica bolesna) wywołana wysiłkiem fizycznym -> zmienna 	kategoryczna
#       wartpść 1 : tak
#       wartość 0 : nie
#
# oldpeak : obniżenie odcinka ST wywołane wysiłkiem fizycznym w stosunku do spoczynku
#
# slope : elektrokardiogram serca przy maksymalnym obciążeniu -> zmienna kategoryczna:
#       wartość 1 : skośne do góry obniżenie ST
#       wartość 2 : poziome obniżenie ST
#       wartość 3 : skośne do dołu obniżenie ST
#
# ca : liczba głównych naczyń (0-3) zabarwionych we fluoroskopii
#
# thal : zaburzenie krwi zwane talasemią (wrodzona niedokrwistość) -> zmienna kategoryczna
#       wartość 3 : normalne
#       wartość 6 : utrwalona wada
#       wartość 7 : odwracalna wada
#
# num : oznaczenie czy osoba jest zdrowa -> zmienna kategoryczna
#      wartość 0 : Osoba zdrowa 
#      wartość 1 - 4 : Osoba z chorobą serca
#
# z powyższego pisu wynika, ze mamy w zbiorze danych 8 zmiennych kategorycznych
# na etapie przeglądania zbioru zweryfikujemy jak dane te zostły wczytane, jakie mają wartości i czy rzeczywiście są to
# zmienne kategoryczne

# Instalacja modulow w spyder
#pip install xgboost
#pip install lightgbm
#pip install scikit-learn
#pip install scikit-posthocs
# pip install sklearn

#instalacja modulow Jupyter
#conda install -c conda-forge xgboost
#conda install -c conda-forge lightgbm

#import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
from scipy.stats import mannwhitneyu   #test Manna Whitneya
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
from scipy.stats import chisquare
import os
from os import path
from sklearn import preprocessing
import pandas as pd
import pylab
import scipy.stats as stats
import scipy.stats as stats
from scipy.stats import shapiro
import scipy.stats as stats
from scipy.stats import normaltest
from statsmodels.stats.diagnostic import lilliefors
import scipy.stats as kstest
from scipy.stats import kstest
from scipy import stats
import scikit_posthocs as sp
from scipy.stats import normaltest
import scipy.stats as stats
import scipy.stats as stats
from scipy.stats import chisquare


# In[265]:


# Zanim przejdziemy do analizy i będziemy tworzyć wykresy
# utworzymy palety kolorów, które zechcemy później wykorzystać do wykresów:
black_grad = ['#100C07', '#3E3B39', '#6D6A6A', '#9B9A9C', '#CAC9CD']
green_grad = ['#006d2c','#2ca25f','#66c2a4','#b2e2e2','#edf8fb']
violet_grad = ['#54278f','#756bb1','#9e9ac8','#cbc9e2', '#f2f0f7']


# In[266]:


# Zobaczmy jak wyglądają stworzone palety
sns.palplot(black_grad)
sns.palplot(green_grad)
sns.palplot(violet_grad)


# In[267]:


# Sprawdzam w jakiej lokalizacji jesteśmy
lokalizacja=os.getcwd()
lokalizacja


# In[268]:


# Sprawdzam czy plik z oryginalnymi danymi znajduje się w  tej samej lokalizacji co plik z kodem
if path.isfile("processed.cleveland.data")==True:
        print("Plik znajduje sie w tym samej lokalizacji co plik z kodem")
else:
        print("Plik z danymi nie znajduje sie w tym samej lokalizacji co plik z kodem")


# In[269]:


dane_odczytu ="\processed.cleveland.data"
dane_zapisu ="\heart.csv"
lokalizacja=os.getcwd()
sciezka_odczytu=lokalizacja+dane_odczytu
sciezka_zapisu=lokalizacja+dane_zapisu 
print("Scieżka odczytu pliku z lokalizacji:")
print(sciezka_odczytu)
print("Scieżka zapisu:")
print(sciezka_zapisu)


# In[270]:


# Ściagamy plik ze strony i dodajemy brakujące nagłówki
dane_oryginalne = pd.read_csv (open(sciezka_odczytu,'r'),header=None
                         , names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak'
                                  ,'slope','ca','thal','num'])
dane_oryginalne.to_csv (open(sciezka_zapisu,'w'), index=None)


# In[271]:


# Spójrzmy na nasze oryginalne dane
with open(sciezka_odczytu,'r') as plik:
    zawartosc=plik.read()
    print(zawartosc)


# In[272]:


# Spójrzmy na nasze zapisane dane
with open(sciezka_zapisu,'r') as plik:
    zawartosc=plik.read()
    print(zawartosc)


# In[273]:


dane_oryginalne.head()


# In[274]:


# podzielmy plik wg przecików
plik_all = "heart.csv"
heart_cleveland = pd.read_csv(plik_all , sep = ",")


# In[275]:


# obejrzyjmy 5 pierwszych wierszy w pliku
heart_cleveland.head()


# In[276]:


# wyświetlimy kształt danych
print("Kształt danych:")
print(heart_cleveland .shape) 


# In[277]:


#jako ze zauważylismy po wstepnej przegladzie pliku ze w danych są znaki specjalne wczytujemy te dane jako braki danych
znaki_specjalne_i_braki_danych= ['~','!','@','#','$','%','^','&','*','(',')','-','=',
                                 '{','}','|',':','"','<','>','/','*','-','+','~_','+',
                                 '{','}','|',':','"','<','>','?']


# In[278]:


#wczytujemy zbiór danych z zamianą znaków specjalnych na na
heart_data_brudne= pd.read_csv("heart.csv", na_values =znaki_specjalne_i_braki_danych)


# In[279]:


#Sprawdzmy ile mamy braków danych
heart_datacos= pd.DataFrame(heart_data_brudne)
heart_datacos.isna().sum().sum()
print("Tyle rekordów ma braki danych w naszym zbiorze:")
print(heart_datacos.isna().sum().sum())
#Mamy 6 wierszy z brakami danych jest to na tyle niewiele że prawdopodobnie usuniemy te wiersze z analizy,
#możemy je także zastąpic wartościami średnymi, modalnymi lub losowymi 


# In[280]:


# Wyświetlmy kilka rekordów z naszego datasetu
heart_data_brudne.head(4)


# In[281]:


#dodajemy kolumne output  na podstawie danych z  num na 0 bez choroby, 1 (wartosci num od 1-4) z choroba serca -zmienna output
heart_data_brudne['output']=np.where(heart_data_brudne['num']<1,0,1)


# In[282]:


# patrze jak wygladaja dane po dodaniu kolumny output
heart_data_brudne.head(4)


# In[283]:


#usuwamy zbedna kolumne num
heart_data_brudne=heart_data_brudne.drop(columns=['num'])


# In[284]:


heart_data_brudne.head(4)


# In[285]:


# wyświetlimy kształt danych
print("Kształt danych:")
print(heart_data_brudne.shape) 


# In[286]:


# zobaczmy wykaz wszystkich kolumn wraz z typem kazdej zmiennej
print("Ogólna informacja o danych:")
print(heart_data_brudne.info())


# In[287]:


# z powyższego widzimy, że większość zmiennych jest typu float64
# część z nich zechcemy później zamienić na category
# iterując po poszczególnych kolumnach, wyświetlę jeszcze liczbę pustych obserwacji dla każdej zmiennej
for column in heart_data_brudne:
    print(column, "- Puste obserwacje: ", heart_data_brudne[column].isnull().sum())


# In[288]:


#mamy 6 pustych obeserwacji, które wcześniej były znakami zapytania
#zobaczmy jak wyglądają rekordy z pustymi ca
heart_data_brudne[heart_data_brudne.ca.isnull()]


# In[289]:


#zobaczmy jak wyglądają rekordy z pustymi thal
heart_data_brudne[heart_data_brudne.thal.isnull()]


# In[290]:


#usuwamy wiersze z pustymi obserwacjami
heart_data = heart_data_brudne.dropna()


# In[291]:


heart_data_5 = heart_data.copy()


# In[292]:


#sprawdzamy kształt danych aktualnych po wyczyszczeniu zbioru z braków
heart_data.shape


# In[293]:


#Sprawzdamy czy pozostaly jakies puste obserwacje
for column in heart_data:
    print(column, "- Puste obserwacje: ", heart_data[column].isnull().sum())


# In[294]:


# widzimy, że zbiór nie zawiera już pustych obserwacji
# sprawdźmy czy są jakieś zdublowane wiersze w naszym zbiorze
duplicates = heart_data.duplicated()
heart_data[duplicates]


# In[295]:


# zanim przejdziemy do dalszej analizy sprawdźmy liczby unikalnych wartości dla każdej zmiennej
heart_data.nunique().sort_values()


# In[296]:


print("sex - płeć - lista wartości:")
print(heart_data["sex"].value_counts())
# dopuszczalne wartości to: 0, 1
# widzimy, że jest OK


# In[297]:


print("fbs - cukier we krwi - lista wartości:")
print(heart_data["fbs"].value_counts().sort_values())
# dopuszczalne wartości to: 0, 1
# widzimy, że jest OK


# In[298]:


print("exang - dusznica bolesna - lista wartości:")
print(heart_data["exang"].value_counts())
# dopuszczalne wartości to: 0, 1
# widzimy, że jest OK


# In[299]:


print("restecg - lista wartości:")
print(heart_data["restecg"].value_counts())
# dopuszczalne wartości to: 0, 1, 2 
# widzimy, że jest OK


# In[300]:


print("slope - ekg przy maks. obciążeniu - lista wartości:")
print(heart_data["slope"].value_counts())
# dopuszczalne wartości to: 1, 2, 3
# widzimy, że jest OK


# In[301]:


print("cp - rodzaj bólu w klatce piersiowej - lista wartości:")
print(heart_data["cp"].value_counts())
# dopuszczalne wartości to: 1, 2, 3, 4
# widzimy, że jest OK


# In[302]:


print("thal - niedokrwistość - lista wartości:")
print(heart_data["thal"].value_counts())
# dopuszczalne wartości to: 3, 6, 7
# widzimy, że jest OK


# In[303]:


print("ca - liczba naczyn - lista wartości:")
print(heart_data["ca"].value_counts())
# dopuszczalne wartości to: 0, 1, 2, 3
# widzimy, że jest OK


# In[304]:


print("output - lista wartości:")
print(heart_data["output"].value_counts())
# dopuszczalne wartości to: 0, 1
# widzimy, że jest już OK , wartości 2,3,4 zostały zamienione na 1 (co oznacza osobę chorą)


# In[305]:


# wyświetlam podstawowe statystyki dla zmiennych ciągłych
heart_data.describe().style.background_gradient(cmap='Purples')


# In[306]:


# zechcę teraz zmienić typ niektórych zmiennych na category
# tworzę nowy zbiór heart_data_1 i wyświetlam kształ nowego zbioru - powinien być taki sam
heart_data_1 = heart_data.copy()
heart_data_copy = heart_data.copy()
print("Kształt danych:")
print(heart_data_1.shape) 


# In[307]:


# teraz bezpiecznie mogę zmienić typ wybranych zmiennych na typ category
heart_data_1["sex"] = heart_data_1["sex"].astype("category")
heart_data_1["fbs"] = heart_data_1["fbs"].astype("category")
heart_data_1["exang"] = heart_data_1["exang"].astype("category")
heart_data_1["output"] = heart_data_1["output"].astype("category")
heart_data_1["restecg"] = heart_data_1["restecg"].astype("category")
heart_data_1["slope"] = heart_data_1["slope"].astype("category")
heart_data_1["cp"] = heart_data_1["cp"].astype("category")
heart_data_1["thal"] = heart_data_1["thal"].astype("category")


# In[308]:


# zobaczmy wykaz wszystkich kolumn wraz z typem kazdej zmiennej
print("Ogólna informacja o danych:")
print(heart_data_1.info())


# In[309]:


heart_data_1.head()


# In[311]:


# Sprawdzenie rozkładu normalnego ceh populacji
####
# Rozkład  graficzny zmiennych za pomocą histogramu
# sprawdzenie graficzne jak kształtują nam się dane
import matplotlib.pyplot as graf
heart_data_2 = heart_data.copy()
dane_niebinarne= pd.DataFrame(heart_data_2 , columns=['age', 'trestbps', 'chol', 'oldpeak','thalach'])
dane_niebinarne.hist(color='purple',figsize=(19,13),layout=(3,5), density=5)
graf.suptitle('Rozkład zmiennych ciągłych')
#zmienne nie wyglądaja aby miały rozklad normalny sprawdzmy wykonujac odpowiednie testy
#widzimy ze zmienna ca to zmienna dyskretna a nie ciągla dlatego nie umieszczamy ja w poniższych wykresach


# In[312]:


#sprawdźmy jak wyglada to z wartościami odstającymuy

boxplot = dane_niebinarne.boxplot(figsize=(18,7), grid=False,flierprops=dict(markersize=14,marker='*', markerfacecolor='purple'))
graf.title('Rozkład zmienych ciągłych',fontsize=15)


# In[313]:


#sprawdzamy czy cechy age, trestbps, chol. oldpeak, thalach charakteryzują się rozkładem normalnym


# In[314]:


# sprawdzmy jak te wygląda wykresy pudełkowe po znormalizowaniu

data_st = dane_niebinarne
scaler = preprocessing.MinMaxScaler()
nazwy_k = dane_niebinarne.columns
danee = scaler.fit_transform(data_st)
normaliz = pd.DataFrame(danee, columns=nazwy_k)
normaliz.head()

boxplot = normaliz.boxplot(figsize=(18,10), grid=False, flierprops=dict(markersize=14,markerfacecolor='red',marker='.'))
graf.title('Rozkład zmienych ciągłych po normalizacji', fontsize=15)
# widzimy ze wszytskie zmienne róznią  sie zasadniczo od siebie pod wzgledem średniej, mediany , kwantyli jak i roztępu


# In[315]:


#sprawdzmy na danych ktore zmienne wykazuje rozklad normalny a które nie na podstawie 
#wykresy kwartyl-kwartyl

stats.probplot(heart_data["age"], dist="norm", plot=pylab)
pylab.title("age")
pylab.show()
stats.probplot(heart_data["trestbps"], dist="norm", plot=pylab)
pylab.title("trestbps")
pylab.show()
stats.probplot(heart_data["chol"], dist="norm", plot=pylab)
pylab.title("chol")
pylab.show()
stats.probplot(heart_data["thalach"], dist="norm", plot=pylab)
pylab.title("thalach")
pylab.show()
stats.probplot(heart_data["oldpeak"], dist="norm", plot=pylab)
pylab.title("oldpeak")
pylab.show()

# Z wykresów wynika ze zmienna age i chol mają najbardziej zbliżony rozkład do normalnego


# In[316]:


# Test d'Agostino
#zobaczmy gradowo jeszcze jak wyglądają rozkłady zmiennych ciągłych na wykresach
for feature in heart_data_1.select_dtypes(include=['float64']):
    sns.distplot(heart_data_1[feature])
    plt.show()
    from scipy.stats import normaltest
    stats, p = normaltest(heart_data_1[feature])
    print(stats, p)
    if p > 0.05:
        print ('Rozkład zmiennej \'' + feature +'\'wygląda na normalny. P-value:', p)
    else:
        print ('Rozkład zmiennej \'' + feature +'\'nie wygląda na normalny. P-value:', p)


# In[317]:


#zobaczmy jak wygląda kurtoza i skośność

heart_data_1.agg(['kurtosis', 'skew']).T


# In[318]:


# test Shapiro Wilk 
from scipy.stats import shapiro
import scipy.stats as stats

statistic, wartosc_p=stats.shapiro(dane_niebinarne["age"])

print("test Shapiro Wilk")
if wartosc_p>0.05:
    print("age-Prawdopodobnie rozkład normalny")
else:
    print("age-Prawdopodobnie nie ma rozkładu normalnego")

#trestbps
statistic, wartosc_p=stats.shapiro(dane_niebinarne["trestbps"])
print(wartosc_p)
if wartosc_p>0.05:
    print("trestbps-Prawdopodobnie rozkład normalny")
else:
    print("trestbps-Prawdopodobnie nie ma rozkładu normalnego")

#chol
statistic, wartosc_p=stats.shapiro(dane_niebinarne["chol"])
if wartosc_p>0.05:
    print("chol-Prawdopodobnie rozkład normalny")
else:
    print("chol-Prawdopodobnie nie ma rozkładu normalnego")
    
#oldpeak
statistic, wartosc_p=stats.shapiro(dane_niebinarne["oldpeak"])
print(wartosc_p)
if wartosc_p>0.05:
    print("oldpeak-Prawdopodobnie rozkład normalny")
else:
    print("oldpeak-Prawdopodobnie nie ma rozkładu normalnego")
    
#thalach
statistic, wartosc_p=stats.shapiro(dane_niebinarne["thalach"])
print(wartosc_p)
if wartosc_p>0.05:
    print("thalach-Prawdopodobnie rozkład normalny")
else:
    print("thalach-Prawdopodobnie nie ma rozkładu normalnego")
    


# In[319]:


# Test d'Aogostino

#age
statistic, wartosc_p=stats.normaltest(dane_niebinarne["age"])
print("test d'Aogostino")
if wartosc_p>0.05:
    print("age-Prawdopodobnie rozkład normalny")
else:
    print("age-Prawdopodobnie nie ma rozkładu normalnego")

#trestbps
statistic, wartosc_p=stats.normaltest(dane_niebinarne["trestbps"])

if wartosc_p>0.05:
    print("trestbps-Prawdopodobnie rozkład normalny")
else:
    print("trestbps-Prawdopodobnie nie ma rozkładu normalnego")
#chol
statistic, wartosc_p=stats.normaltest(dane_niebinarne["chol"])

if wartosc_p>0.05:
    print("chol-Prawdopodobnie rozkład normalny")
else:
    print("chol-Prawdopodobnie nie ma rozkładu normalnego")
#oldpeak
statistic, wartosc_p=stats.normaltest(dane_niebinarne["oldpeak"])

if wartosc_p>0.05:
    print("oldpeak-Prawdopodobnie rozkład normalny")
else:
    print("oldpeak-Prawdopodobnie nie ma rozkładu normalnego")
#thalach
statistic, wartosc_p=stats.normaltest(dane_niebinarne["thalach"])

if wartosc_p>0.05:
    print("thalach-Prawdopodobnie rozkład normalny")
else:
    print("thalach-Prawdopodobnie nie ma rozkładu normalnego")


# In[320]:


# test Jarque-Bera 

print("test Jarque-Bera")
dane_niebinarne
#age
statistic, wartosc_p=stats.jarque_bera(dane_niebinarne["age"])

if wartosc_p>0.05:
    print("age-Prawdopodobnie rozkład normalny")
else:
    print("age-Prawdopodobnie nie ma rozkładu normalnego")
#trestbps
statistic, wartosc_p=stats.jarque_bera(dane_niebinarne["trestbps"])

if wartosc_p>0.05:
    print("trestbps-Prawdopodobnie rozkład normalny")
else:
    print("trestbps-Prawdopodobnie nie ma rozkładu normalnego")
#chol
statistic, wartosc_p=stats.jarque_bera(dane_niebinarne["chol"])

if wartosc_p>0.05:
    print("chol-Prawdopodobnie rozkład normalny")
else:
    print("chol-Prawdopodobnie nie ma rozkładu normalnego")
#oldpeak
statistic, wartosc_p=stats.jarque_bera(dane_niebinarne["oldpeak"])

if wartosc_p>0.05:
    print("oldpeak-Prawdopodobnie rozkład normalny")
else:
    print("oldpeak-Prawdopodobnie nie ma rozkładu normalnego")
#thalach
statistic, wartosc_p=stats.jarque_bera(dane_niebinarne["thalach"])

if wartosc_p>0.05:
    print("thalach-Prawdopodobnie rozkład normalny")
else:
    print("thalach-Prawdopodobnie nie ma rozkładu normalnego")


# In[321]:


#test lilliefors
print("test lilliefors")
statistic, wartosc_p=lilliefors(dane_niebinarne["age"])
#chol
statistic, wartosc_p=lilliefors(dane_niebinarne["chol"])
if wartosc_p>0.05:
    print("chol-Prawdopodobnie rozkład normalny")
else:
    print("chol-Prawdopodobnie nie ma rozkładu normalnego")
#age
if wartosc_p>0.05:
    print("age-Prawdopodobnie rozkład normalny")
else:
    print("age-Prawdopodobnie nie ma rozkładu normalnego")
#trestbps

statistic, wartosc_p=lilliefors(dane_niebinarne["trestbps"])
if wartosc_p>0.05:
    print("trestbps-Prawdopodobnie rozkład normalny")
else:
    print("trestbps-Prawdopodobnie nie ma rozkładu normalnego")
#oldpeak
statistic, wartosc_p=lilliefors(dane_niebinarne["oldpeak"])
if wartosc_p>0.05:
    print("oldpeak-Prawdopodobnie rozkład normalny")
else:
    print("oldpeak-Prawdopodobnie nie ma rozkładu normalnego")
#thalach
statistic, wartosc_p=lilliefors(dane_niebinarne["thalach"])
if wartosc_p>0.05:
    print("thalach-Prawdopodobnie rozkład normalny")
else:
    print("thalach-Prawdopodobnie nie ma rozkładu normalnego")
    


# In[322]:


# test chisquare

print("test chisquare")
from scipy.stats import chisquare
statistic, wartosc_p=chisquare(dane_niebinarne["age"])

if wartosc_p> 0.05:
    print("age-Prawdopodobnie rozkład normalny")
else:
    print("age-Prawdopodobnie nie ma rozkładu normalnego")
#trestbps
statistic, wartosc_p=stats.chisquare(dane_niebinarne["trestbps"])

if wartosc_p>0.05:
    print("trestbps-Prawdopodobnie rozkład normalny")
else:
    print("trestbps-Prawdopodobnie nie ma rozkładu normalnego")
#chol
statistic, wartosc_p=stats.chisquare(dane_niebinarne["chol"])

if wartosc_p>0.05:
    print("chol-Prawdopodobnie rozkład normalny")
else:
    print("chol-Prawdopodobnie nie ma rozkładu normalnego")
#oldpeak
statistic, wartosc_p=stats.chisquare(dane_niebinarne["oldpeak"])

if wartosc_p>0.05:
    print("oldpeak-Prawdopodobnie rozkład normalny")
else:
    print("oldpeak-Prawdopodobnie nie ma rozkładu normalnego")
#thalach
statistic, wartosc_p=stats.chisquare(dane_niebinarne["thalach"])

if wartosc_p>0.05:
    print("thalach-Prawdopodobnie rozkład normalny")
else:
    print("thalach-Prawdopodobnie nie ma rozkładu normalnego")  


# In[323]:


# test Kołmogorowa-Smirnova
print("test Kołmogorowa-Smirnova")
statistic, wartosc_p=kstest(dane_niebinarne["age"],'norm')

if wartosc_p> 0.05:
    print("age-Prawdopodobnie rozkład normalny")
else:
    print("age-Prawdopodobnie nie ma rozkładu normalnego")
#trestbps
statistic, wartosc_p=kstest(dane_niebinarne["trestbps"],'norm')

if wartosc_p>0.05:
    print("trestbps-Prawdopodobnie rozkład normalny")
else:
    print("trestbps-Prawdopodobnie nie ma rozkładu normalnego")
#chol
statistic, wartosc_p=kstest(dane_niebinarne["chol"],'norm')

if wartosc_p>0.05:
    print("chol-Prawdopodobnie rozkład normalny")
else:
    print("chol-Prawdopodobnie nie ma rozkładu normalnego")
#oldpeak
statistic, wartosc_p=kstest(dane_niebinarne["oldpeak"],'norm')

if wartosc_p>0.05:
    print("oldpeak-Prawdopodobnie rozkład normalny")
else:
    print("oldpeak-Prawdopodobnie nie ma rozkładu normalnego")
#thalach
statistic, wartosc_p=kstest(dane_niebinarne["thalach"],'norm')

if wartosc_p>0.05:
    print("thalach-Prawdopodobnie rozkład normalny")
else:
    print("thalach-Prawdopodobnie nie ma rozkładu normalnego")  


# In[325]:



#zmieniam nazwy parametrow output aby moc zinterpretowac poprawnie dane w testach posthoc
zbior = heart_data_1
nazwycp={3:'ból nie związany z dusznicą bolesną',2:'nietypowa dusznica bolesna',1:'typowa dusznica bolesna',4:'bezobjawowy'}
nazwyrestecg={0:'prawidłowe',1:'z nieprawidłowościami załamka ST-T',2:'wyraźny przerost lewej komory - stan ciężki' }
nazwyslope={1:'skośne do góry obniżenie ST',2:'poziome obniżenie ST',3:'skośne do dołu obniżenie ST'}
nazwythal={3:'normalne',6:'utrwalona wada', 7:'odwracalna wada'}

zbior['cp_zmiana']=zbior['cp'].replace(nazwycp)
zbior['restecg_zmiana']=zbior['restecg'].replace(nazwyrestecg)
zbior['slope_zmiana']=zbior['slope'].replace(nazwyslope)
zbior['thal_zmiana']=zbior['thal'].replace(nazwythal)

zbior.head()


# In[326]:


# test Kruskala Wallisa 
#Sprawdzenie dla danych porządkowych czy osoby chore i zdrowy roznicuja poszczegolne zmiennie
#cp
cp_o_zdrowa = heart_data['cp'][heart_data['output']==0]
cp_o_chora = heart_data['cp'][heart_data['output']==1]


statistic, wartosc_p=stats.kruskal(cp_o_zdrowa,cp_o_chora)
print(wartosc_p)
if wartosc_p<0.05:
    print("cp-Prawdopodobnie istotnie  roznicuje osoby chore od zdrowych ze względu na tą zmienną")
else:
    print("cp-Prawdopodobnie istotnie  nie roznicuje osoby chore od zdrowych  ze względu na tą zmienną") 


# In[327]:


sp.posthoc_dunn(zbior,'output', 'cp_zmiana')
# istornie nie roznia się miedzy osobą osoby chore i zdrowe miedzy grupami:
#typowa dusznica bolesna a bol niezwiazany z dusznica bolesna
#typowa dusznica bolesna a nietypowa dusznica bolesna
# ból nie związany z dusznicą bolesną i nietypowa dusznica bolesna
# w pozostałych podzbiorach istotna


# In[328]:


#restecg

restecg_o_zdrowa = heart_data['restecg'][heart_data['output']==0]
restecg_o_chora = heart_data['restecg'][heart_data['output']==1]


statistic, wartosc_p=stats.kruskal(restecg_o_zdrowa,restecg_o_chora)
print(wartosc_p)
if wartosc_p<0.05:
    print("restecg-Prawdopodobnie istotnie roznicuje osoby chore od zdrowych ze względu na tą zmienną")
else:
    print("restecg-Prawdopodobnie istotnie  nie roznicuje osoby chore od zdrowych  ze względu na tą zmienną")  


# In[329]:


sp.posthoc_dunn(zbior,'output', 'restecg_zmiana')
# istornie nie roznia się miedzy osobą osoby chore i zdrowe miedzy grupamio
#z nieprawidłowościami załamka ST-T	 i osobowami z prawidlowym zalamkiem ST
#wyraźny przerost lewej komory - stan ciężki i z nieprawidłowościami załamka ST-T	
# istotnie roznia się tylko grupa wyraźny przerost lewej komory - stan ciężki i prawidłowe ST


# In[330]:


#slope

slope_o_zdrowa = heart_data['slope'][heart_data['output']==0]
slope_o_chora = heart_data['slope'][heart_data['output']==1]


statistic, wartosc_p=stats.kruskal(slope_o_zdrowa,slope_o_chora)
print(wartosc_p)
if wartosc_p<0.05:
    print("slope-Prawdopodobnie istotnie roznicuje osoby chore od zdrowych ze względu na tą zmienną")
else:
    print("slope-Prawdopodobnie istotnie  nie roznicuje osoby chore od zdrowych  ze względu na tą zmienną")      


# In[331]:


sp.posthoc_dunn(zbior,'output', 'slope_zmiana')

# In[332]:


#thal

thal_o_zdrowa = heart_data['thal'][heart_data['output']==0]
thal_o_chora = heart_data['thal'][heart_data['output']==1]


statistic, wartosc_p=stats.kruskal(thal_o_zdrowa,thal_o_chora)
print(wartosc_p)
if wartosc_p<0.05:
    print("thal-Prawdopodobnie istotnie roznicuje osoby chore od zdrowych ze względu na tą zmienną")
else:
    print("thal-Prawdopodobnie istotnie  nie roznicuje osoby chore od zdrowych  ze względu na tą zmienną") 


# In[333]:


sp.posthoc_dunn(zbior,'output', 'thal_zmiana')
# istornie nieroznia sie miedzy sobą grupa osob chorych z zdrowym z utrwalona wada i odwracalna wada


# In[334]:


#test chi kwadrat zmienna sex
import scipy.stats as stats
from scipy.stats import chi2_contingency

print("test chi kwadrat")
dane_allsex = pd.DataFrame(heart_data, columns=['sex','output'])

# Tworzymy table kontyngencji
dane_tabela=pd.crosstab(index=dane_allsex['sex'], columns=dane_allsex['output'])
print('tabela kontyngencji')
print(dane_tabela)

chi2, pwartosc, dof, ex = chi2_contingency(dane_tabela)
print('pvalue :' ,pwartosc)

if pwartosc<= 0.05:
    print('Zmienna sex nie różnicuje między sobą osoby chore i zdrowe')
else:
    print('Zmienna sex istotnie różnicuje między sobą osoby chore i zdrowe')


# In[335]:


#test chi kwadrat zmienna fbs
dane_allfbs = pd.DataFrame(heart_data, columns=['fbs','output'])

# Tworzymy table kontyngencji
dane_tabela=pd.crosstab(index=dane_allfbs['fbs'], columns=dane_allfbs['output'])
print('tabela kontyngencji')
print(dane_tabela)

chi2, pwartosc, dof, ex = chi2_contingency(dane_tabela)
print('pvalue :' ,pwartosc)

if pwartosc<= 0.05:
    print('Zmienna fbs nie różnicuje między sobą osoby chore i zdrowe')
else:
    print('Zmienna fbs istotnie różnicuje między sobą osoby chore i zdrowe')


# In[336]:


#test chi kwadrat zmienna exang
dane_allexang = pd.DataFrame(heart_data, columns=['exang','output'])

# Tworzymy table kontyngencji
dane_tabela=pd.crosstab(index=dane_allexang['exang'], columns=dane_allexang['output'])
print('tabela kontyngencji')
print(dane_tabela)

chi2, pwartosc, dof, ex = chi2_contingency(dane_tabela)
print('pvalue :' ,pwartosc)

if pwartosc<= 0.05:
    print('Zmienna exang nie różnicuje między sobą osoby chore i zdrowe')
else:
    print('Zmienna exang istotnie różnicuje między sobą osoby chore i zdrowe')


# In[337]:


#Wybieram do testu Manna Whitmanna kolumnę restecg -"spoczynkowe wyniki elektrokardiograficzne EKG pacjenta" osobno dla zdrowych i chorych
restecg = pd.DataFrame(heart_data, columns=['restecg', 'output'])
#print (restecg)
restecg_chory =  heart_data['restecg'][heart_data['output']==1]
restecg_chory_1= restecg_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
#print(restecg_chory)
restecg_zdrowy = heart_data['restecg'][heart_data['output']==0]
restecg_zdrowy_1= restecg_zdrowy.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
#print(restecg_zdrowy)


# In[338]:


print(restecg_chory_1.shape)
restecg_chory_1.describe()


# In[339]:


print(restecg_zdrowy_1.shape)
restecg_chory_1.describe()


# In[340]:


# test dla rectecg, posortowane wartości
stats.mannwhitneyu(restecg_zdrowy_1, restecg_chory_1, alternative='two-sided')


# In[341]:


# Kontynuuję na danej slope - ekg przy maks. obciążeniu
slope = pd.DataFrame(heart_data, columns=['slope', 'output'])
slope_chory =  heart_data['slope'][heart_data['output']==1]
slope_chory_1= slope_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
slope_zdrowy = heart_data['slope'][heart_data['output']==0]
slope_zdrowy_1= slope_zdrowy.sort_values()


# In[342]:


print(slope_chory_1.shape)
slope_chory_1.describe()


# In[343]:


print(slope_zdrowy_1.shape)
slope_zdrowy_1.describe()


# In[344]:


stats.mannwhitneyu(slope_zdrowy_1, slope_chory_1, alternative='two-sided')


# In[345]:


#Wybieram do testu Manna Whitmanna kolumnę cp - "rodzaj bólu w klatce piersiowej", osobno dla zdrowych i chorych
cp = pd.DataFrame(heart_data, columns=['cp', 'output'])
cp_chory = heart_data['cp'][heart_data['output']==1]
cp_chory_1= cp_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
cp_zdrowy = heart_data['cp'][heart_data['output']==0]
cp_zdrowy_1= cp_zdrowy.sort_values()


# In[346]:


print(cp_zdrowy.shape)
#cp_zdrowy_1.describe()
print(cp_chory_1.shape)
#cp_chory_1.describe()


# In[347]:


#Sprawdzam, czy dana cp - "rodzaj bólu w klatce piersiowej" różnicuje grupy zdrowych i chorych. W tym celu wykonuję test Manna whitmana. 
stats.mannwhitneyu(cp_zdrowy_1, cp_chory_1, alternative='two-sided')


# In[348]:


#Wybieram do testu Manna Whitmanna kolumnę thal - niedokrwistość, osobno dla zdrowych i chorych
thal = pd.DataFrame(heart_data, columns=['thal', 'output'])
thal_chory = heart_data['thal'][heart_data['output']==1]
thal_chory_1= thal_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
thal_zdrowy = heart_data['thal'][heart_data['output']==0]
thal_zdrowy_1= thal_zdrowy.sort_values()


# In[349]:


print(thal_chory_1.shape)
#thal_chory_1.describe()
print(thal_zdrowy_1.shape)
#thal_zdrowy_1.describe()


# In[350]:


stats.mannwhitneyu(thal_zdrowy_1, thal_chory_1, alternative='two-sided')


# In[351]:


#Wybieram do testu Manna Whitmanna kolumnę ca - liczba naczyn, osobno dla zdrowych i chorych
ca = pd.DataFrame(heart_data, columns=['ca', 'output'])
ca_chory = heart_data['ca'][heart_data['output']==1]
ca_chory_1= ca_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
ca_zdrowy = heart_data['ca'][heart_data['output']==0]
ca_zdrowy_1= ca_zdrowy.sort_values()


# In[352]:


print(ca_chory_1.shape)
#ca_chory_1.describe()
print(ca_zdrowy_1.shape)
#ca_zdrowy_1.describe()


# In[353]:


stats.mannwhitneyu(ca_zdrowy_1, ca_chory_1, alternative='two-sided')


# In[354]:


#przenaliuzję zmienną trestbps - spoczynkowe ciśnienie krwi (w mm Hg) przy przyjęciu do szpitala
trestbps = pd.DataFrame(heart_data, columns=['trestbps', 'output'])
trestbps_chory = heart_data['trestbps'][heart_data['output']==1]
trestbps_chory_1= trestbps_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
trestbps_zdrowy = heart_data['trestbps'][heart_data['output']==0]
trestbps_zdrowy_1= trestbps_zdrowy.sort_values()


# In[355]:


print(trestbps_chory_1.shape)
print(trestbps_zdrowy_1.shape)


# In[356]:


stats.mannwhitneyu(trestbps_zdrowy_1, trestbps_chory_1, alternative='two-sided')


# In[357]:


#przenaliuzję zmienną chol -  poziom cholesterolu pacjenta w mg/dl przy przyjęciu do szpitala
chol = pd.DataFrame(heart_data, columns=['chol', 'output'])
chol_chory = heart_data['chol'][heart_data['output']==1]
chol_chory_1= chol_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
chol_zdrowy = heart_data['chol'][heart_data['output']==0]
chol_zdrowy_1= chol_zdrowy.sort_values()


# In[358]:


print(chol_chory_1.shape)
print(chol_zdrowy_1.shape)


# In[359]:


chol_zdr= heart_data['restecg'][heart_data['output']==0]
chol_chory= heart_data['restecg'][heart_data['output']==1]
stats.mannwhitneyu(chol_zdr, chol_chory, alternative='two-sided')


# In[360]:


#przenaliuzję zmienną thalach: osiągnięte maksymalne tętno przy przyjęciu do szpitala
thalach = pd.DataFrame(heart_data, columns=['thalach', 'output'])
thalach_chory =  heart_data['thalach'][heart_data['output']==1]
thalach_chory_1= thalach_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
thalach_zdrowy =  heart_data['thalach'][heart_data['output']==0]
thalach_zdrowy_1= thalach_zdrowy.sort_values()


# In[361]:


thalach_chory_1


# In[362]:


print(thalach_chory_1.shape)
print(thalach_zdrowy_1.shape)


# In[363]:


stats.mannwhitneyu(thalach_zdrowy_1, thalach_chory_1, alternative='two-sided')


# In[364]:


#przenaliuzję zmienną age: wiek pacjenta w latach
age = pd.DataFrame(heart_data, columns=['age', 'output'])
age_chory =  heart_data['age'][heart_data['output']==1]
age_chory_1= age_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
age_zdrowy = heart_data['age'][heart_data['output']==0]
age_zdrowy_1= age_zdrowy.sort_values()


# In[365]:


print(age_chory_1.shape)
print(age_zdrowy_1.shape)


# In[366]:


stats.mannwhitneyu(age_zdrowy_1, age_chory_1, alternative='two-sided')


# In[367]:


#przenaliuzję zmienną oldpeak
oldpeak = pd.DataFrame(heart_data, columns=['oldpeak', 'output'])
oldpeak_chory = heart_data['oldpeak'][heart_data['output']==1]
oldpeak_chory_1= oldpeak_chory.sort_values() #sortuję watości rosnąco - ma wpływ na poprawnośc testu
oldpeak_zdrowy = heart_data['oldpeak'][heart_data['output']==0]
oldpeak_zdrowy_1= oldpeak_zdrowy.sort_values()


# In[368]:


print(oldpeak_chory_1.shape)
print(oldpeak_zdrowy_1.shape)


# In[369]:



stats.mannwhitneyu(oldpeak_zdrowy_1, oldpeak_chory_1, alternative='two-sided')


# In[370]:


# Test LEVENE'a o jednorodności wariancji 
from scipy.stats import levene
agee_zdr= heart_data['age'][heart_data['output']==0]
agee_chor= heart_data['age'][heart_data['output']==1]
wstatystyka, pwartosc = levene(agee_zdr, agee_chor)
print(pwartosc)
if pwartosc<0.05:
    print("podane grupy nie mają równej wariancji")
else:
    print("prawdopobnie grupy mają równą wariancje")  
# HO: jest równa wariancja
# Ha: brak rownej wariancji
#Wariancje miedzy tymi grupami średnio są różne  , ponieważ p.value  jest mniejszeod 0.05 
#stąd  możemy odrzucic hipoteze zerowa na rzecz hipotezy alternatywnej mówiącej o roznorodnosci wariancji 


# In[371]:


# Test LEVENE'a o jednorodności wariancji
from scipy.stats import levene
chol_zdr= heart_data['chol'][heart_data['output']==0]
chol_chor= heart_data['chol'][heart_data['output']==1]
wstatystyka, pwartosc = levene(chol_zdr, chol_chor)
print(pwartosc)
if pwartosc<0.05:
    print("podane grupy nie mają równej wariancji")
else:
    print("prawdopodobnie grupy mają równą wariancje")  
# HO: jest równa wariancja
# Ha: brak rownej wariancji
#Wariancje miedzy tymi grupami średnio są rowne , ponieważ p.value  jest większe od 0.05 
#stąd nie możemy odrzucic hipoteze zerowa mówiąco o równej wariancji tych grup     


# In[372]:


#Sprawdżmy na wykresie jak kształtuje się rozklady osob chorych i zdrowych ze względu na wiek i cholesterol
age_o_zdrowa = heart_data['age'][heart_data['output']==0]
age_o_chora = heart_data['age'][heart_data['output']==1]
chole_o_zdrowa = heart_data['chol'][heart_data['output']==0]
chole_o_chora = heart_data['chol'][heart_data['output']==1]

all1 = [age_o_zdrowa , age_o_chora ,chole_o_zdrowa,chole_o_chora]
obraz = plt.figure(figsize =(12, 12))
sr = obraz.add_axes([0, 0, 1, 1])
plt.show(sr.boxplot(all1))
#średnia dla wieku 1, 2 boxplot na podstawie rysunku rózni sie miedzy sobą dla osob zdrowych i chorych
#tak samo średnia dla cholesterolu rozni sie miedzy osoba chorymi i zdrowymi 
#Wariancja patrząc na wielkośc pudełka w wykresie pudełkowym , potwierdza ze wariancja osób grupy chorych i grupy zdrowych
#ze względu na wiek średnio jjest różna ze względu na wiek i równa ze względu na cholestrol


# In[373]:


# test t-studenta
age_zdr= heart_data['age'][heart_data['output']==0]
age_chor= heart_data['age'][heart_data['output']==1]
stats.ttest_ind(age_zdr,age_chor, equal_var=False, alternative='two-sided')
#h0 srednie sa rowne
#wiek róznicuje choroby zdrowe i chore poniewaz p value <0.05


# In[374]:


# test t-studenta
chol_zdr= heart_data['chol'][heart_data['output']==0]
chol_chor= heart_data['chol'][heart_data['output']==1]
stats.ttest_ind(chol_zdr,chol_chor, equal_var=True, alternative='two-sided')
#h0 srednie sa rowne
#cholesterol nie  róznicuje choroby zdrowe i chore poniewaz p value >0.05 i możemy zalożyć średnia wartość tych podgrup jest równa


# In[375]:


# wyświetlam podstawowe statystyki dla zmiennych ciągłych
heart_data_1.describe().style.background_gradient(cmap='Purples')


# In[376]:


# zobaczmy jak zmienne ciągłe są skorelowane 
heart_data_1.corr().style.background_gradient(cmap='summer')

#Test istotności współczynników korelacji opiera się na założeniu o normalności rozkładu wartości resztowych
#(odchyleń od linii regresji) zmiennej y, oraz o równości wariancji wartości resztowych dla 
#wszystkich wartości zmiennej niezależnej x. 
#Jeżeli próbka liczy 100 lub więcej, wówczas założeniem o normalności nie należy się praktycznie przejmować. 
#Trzeba pamiętać o obserwacjach odstających, które mają duży wpływ na nachylenie linii regresji czyli
# na wartość współczynnika korelacji. Nie mozna wyciągać wniosków tylko na podstawie wartości współczynnika korelacji
# age i thalach (max tętno) są lekko ze sobą skorelowane


# In[377]:


# wyliczenie udziału procentowego badanych kobiet i mężczyzn
print('{} % to badani mężczyźni'.format(heart_data_1['sex'][heart_data_1['sex']==1].count()*100/heart_data_1['sex'].count()))
print('{} % to badane kobiety'.format(heart_data_1['sex'][heart_data_1['sex']==0].count()*100/heart_data_1['sex'].count()))

# z poniższej statystyki wynika, że większość badanych to mężczyźni


# In[378]:


# obejrzyjmy jeszcze jak wygląda rozkład płci na wykresie

colors=green_grad[2:4]
labels=['Mężczyźni', 'Kobiety']
order=heart_data_1['sex'].value_counts().index
# Rozmiar wykresów
plt.figure(figsize=(10, 5))
plt.suptitle('Rozkład płci', fontweight='heavy', 
             fontsize='10', fontfamily='sans-serif', color=black_grad[0])

# Wykres kołowy
plt.subplot(1, 2, 1)
plt.title('Wykres kołowy', fontweight='bold', fontsize=10,
          fontfamily='sans-serif', color=black_grad[0])
plt.pie(heart_data_1['sex'].value_counts(), labels=labels, colors=colors, pctdistance=0.7,
        autopct='%.2f%%', wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]),
        textprops={'fontsize':10})
centre=plt.Circle((0, 0), 0.25, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre)

# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=10, 
          fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='sex', data=heart_data_1, palette=colors, order=order,
                   edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=8, 
             bbox=dict(facecolor='none', edgecolor=black_grad[0], 
                       linewidth=0.25, boxstyle='round'))

plt.xlabel('Płeć', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])

plt.xticks([0, 1], labels)
plt.grid(axis='y', alpha=0.4)
countplt

# podsumowanie
print('*' * 55)
print('\033[1m'+'Łączna liczba badanych pacjentów w podziale na płeć:'+'\033[0m')
print('*' * 55)


# In[379]:


# obejrzyjmy jeszcze jak wygląda rozkład fbs na wykresie

colors=green_grad[2:4]
labels=['Poniżej 120 mg/dl','Powyżej 120 mg/dl']
order=heart_data_1['fbs'].value_counts().index

# Rozmiar wykresów
plt.figure(figsize=(10, 5))
plt.suptitle('Rozkład fbs (poziom cukru we krwi)', fontweight='heavy', 
             fontsize='10', fontfamily='sans-serif', color=black_grad[0])

# Wykres kołowy
plt.subplot(1, 2, 1)
plt.title('Wykres kołowy', fontweight='bold', fontsize=10,
          fontfamily='sans-serif', color=black_grad[0])
plt.pie(heart_data_1['fbs'].value_counts(), labels=labels, colors=colors, pctdistance=0.7,
        autopct='%.2f%%', wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]),
        textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.25, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre)

# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=10, 
          fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='fbs', data=heart_data_1, palette=colors, order=order,
                   edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=8, 
             bbox=dict(facecolor='none', edgecolor=black_grad[0], 
                       linewidth=0.25, boxstyle='round'))

plt.xlabel('fbs', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=violet_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
           color=violet_grad[1])

plt.xticks([0, 1], labels)
plt.grid(axis='y', alpha=0.4)
countplt

# podsumowanie
print('*' * 85)
print('\033[1m'+'Łączna liczba badanych pacjentów w podziale na fbs (poziom cukru powyżej 120 mg/dl):'+'\033[0m')
print('*' * 85)
# widzimy, że większość badanych osób miała poziom cukru prawidłowy tj. poniżej 120 mg/dl


# In[380]:


# obejrzyjmy jeszcze jak wygląda rozkład restecg (spoczynkowe EKG pacjenta) na wykresie, gdzie:
# wartość 0: prawidłowe
# wartość 1: z nieprawidłowościami
# wartość 2: stan ciężki

colors=green_grad[1:4]
labels=['prawidłowe', 'stan ciężki','niepraw.']
order=heart_data_1['restecg'].value_counts().index

# Rozmiar wykresów
plt.figure(figsize=(10, 5))
plt.suptitle('Rozkład restecg (spoczynkowe EKG pacjenta)', fontweight='heavy', 
             fontsize='10', fontfamily='sans-serif', color=black_grad[0])

# Wykres kołowy
plt.subplot(1, 2, 1)
plt.title('Wykres kołowy', fontweight='bold', fontsize=10,
          fontfamily='sans-serif', color=black_grad[0])

plt.pie(heart_data_1['restecg'].value_counts(), labels=labels, colors=colors, pctdistance=0.7,
        autopct='%.2f%%', wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]),
        textprops={'fontsize':8})

centre=plt.Circle((0, 0), 0.55, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre)

# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=10, 
          fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='restecg', data=heart_data_1, palette=colors, order=order,
                   edgecolor=black_grad[2], alpha=0.85)

for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+3.25,rect.get_height(), 
             horizontalalignment='center', fontsize=8, 
             bbox=dict(facecolor='none', edgecolor=black_grad[0], 
                       linewidth=0.25, boxstyle='round'))
    
plt.xlabel('restecg', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])

#plt.xticks([0, 1], labels)
plt.grid(axis='y', alpha=0.4)
countplt

# podsumowanie
print('*' * 85)
print('\033[1m'+'Łączna liczba badanych pacjentów w podziale na restecg (spoczynkowe EKG pacjenta):'+'\033[0m')
print('*' * 85)
# z poniższych wykresó widzimy, że domunujące wyniki to osoby z prawidłowym EKG lub w stanie cięzkim
# jedynie kilka obserwacji dotyczy osób z nieprawidłowościami EKG


# In[381]:



# obejrzyjmy jeszcze jak wygląda rozkład exang (występowania dławicy piersiowej) na wykresie
#       wartpść 1 : tak
#       wartość 0 : nie

colors=green_grad[2:4]
labels=['Brak','Występuje']
order=heart_data_1['exang'].value_counts().index

# Rozmiar wykresów
plt.figure(figsize=(10, 5))
plt.suptitle('Rozkład exang (występowanie dławicy piersiowej)', fontweight='heavy', 
             fontsize='10', fontfamily='sans-serif', color=black_grad[0])

# Wykres kołowy
plt.subplot(1, 2, 1)
plt.title('Wykres kołowy', fontweight='bold', fontsize=10,
          fontfamily='sans-serif', color=black_grad[0])
plt.pie(heart_data_1['exang'].value_counts(), labels=labels, colors=colors, pctdistance=0.7,
        autopct='%.2f%%', wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]),
        textprops={'fontsize':10})
centre=plt.Circle((0, 0), 0.25, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre)

# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=10, 
          fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='exang', data=heart_data_1, palette=colors, order=order,
                   edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=8, 
             bbox=dict(facecolor='none', edgecolor=black_grad[0], 
                       linewidth=0.25, boxstyle='round'))

plt.xlabel('exang', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])

plt.xticks([0, 1], labels)
plt.grid(axis='y', alpha=0.4)
countplt

# podsumowanie
print('*' * 90)
print('\033[1m'+'Łączna liczba badanych pacjentów w podziale na exang (występowanie dławicy piersiowej):'+'\033[0m')
print('*' * 90)
# widzimy, że u większości badanych osób nie występuje dławica piersiowa


# In[382]:


# obejrzyjmy jeszcze jak wygląda rozkład slope (EKG przy maksymalnym obciążeniu) na wykresie, gdzie:
#       wartość 1 : skośne do góry obniżenie ST
#       wartość 2 : poziome obniżenie ST
#       wartość 3 : skośne do dołu obniżenie ST

colors=green_grad[1:4]
labels=['sk.góra', 'poziome','sk.dół']
order=heart_data_1['slope'].value_counts().index

# Rozmiar wykresów
plt.figure(figsize=(10, 5))
plt.suptitle('Rozkład slope (EKG przy maksymalnym obciążeniu)', fontweight='heavy', 
             fontsize='10', fontfamily='sans-serif', color=black_grad[0])

# Wykres kołowy
plt.subplot(1, 2, 1)
plt.title('Wykres kołowy', fontweight='bold', fontsize=10,
          fontfamily='sans-serif', color=black_grad[0])

plt.pie(heart_data_1['slope'].value_counts(), labels=labels, colors=colors, pctdistance=0.7,
        autopct='%.2f%%', wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]),
        textprops={'fontsize':10})

centre=plt.Circle((0, 0), 0.25, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre)

# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title('Histogram', fontweight='bold', fontsize=10, 
          fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='slope', data=heart_data_1, palette=colors, order=order,
                   edgecolor=black_grad[2], alpha=0.85)

for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+4.25,rect.get_height(), 
             horizontalalignment='center', fontsize=8, 
             bbox=dict(facecolor='none', edgecolor=black_grad[0], 
                       linewidth=0.25, boxstyle='round'))
    
plt.xlabel('slope', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=10, fontfamily='sans-serif', 
           color=violet_grad[1])

#plt.xticks([0, 1], labels)
plt.grid(axis='y', alpha=0.4)
countplt

# podsumowanie
print('*' * 88)
print('\033[1m'+'Łączna liczba badanych pacjentów w podziale na slope (EKG przy maksymalnym obciążeniu) :'+'\033[0m')
print('*' * 88)
# z poniższych wykresów widzimy, że domunujące wyniki to osoby ze skośnym do góry obsniżeniem ST lub z poziomym
# mała część obserwacji dotyczy osób ze skośnym do dołu obniżeniem ST


# In[383]:


# zobaczmy jak wygląda podział osób chorych i zdrowych ze względu na płeć
plt.figure(figsize=(5,5))
ax=sns.countplot(x='output', y=None, hue='sex', data=heart_data_1, order=[1,0]
                 , color=green_grad[2:4], palette='summer_r', saturation=0.75)
ax.set_xticklabels(['Pacjent z chorobą serca','Osoba zdrowa'])

# podsumowanie
print('*' * 50)
print('\033[1m'+'Podział na osoby chore i zdrowe ze względu na płeć'+'\033[0m')
print('*' * 50)


# In[384]:


# zobaczmy jak kształtują się zmienne kategoryczne
# w zależności od płci i występowania chorób serca

for column in ['cp', 'restecg','fbs','exang','slope','thal']:
    plt.figure(figsize=(5,5))
    ax=sns.catplot(x='output',data=heart_data_1,hue=column,col='sex',kind='count',order=[1,0], palette="summer")
    ax.set_xticklabels(['Pacjent z chorobą serca','Osoba zdrowa'])


# In[385]:


# policzmy ile wynosi średni choleterol w podziale na płeć
print('mężczyźni-{}'.format(heart_data_1['chol'][heart_data_1['sex']==1].mean()))
print('kobiety-{}'.format(heart_data_1['chol'][heart_data_1['sex']==0].mean()))


# In[386]:


# zobaczmy jeszcze jak kształtuje się minimalna i maksymalna wartość cholesterolu w podziale na płeć
heart_data_1.groupby('sex')['chol'].agg([min, max])


# In[387]:


heart_data_1.apply(pd.Series.value_counts).style.background_gradient(cmap='summer')
