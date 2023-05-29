import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import math as m
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import sleep
data = pd.read_excel('actors.xls')
data


print('Кол-во мужчин:',data[data['sex']=='M']['humanID'].nunique())
print('Кол-во женщин:',data[data['sex']=='F']['humanID'].nunique())


corona = data[data['isCoronavirus'] == 'Yes']
print("Людей, болевших коронавирусом:",corona['humanID'].nunique())


lungInjury = data[data['isLungInjury'] == 'Yes']
print("Людей с поражением легких:",lungInjury['humanID'].nunique())


#Функция создает таблицу с вычесленными векторами и их изменением
def makeVectorTableOther(dfdata):
    n = 10 #Для скользящего среднего
    leng1 = np.sqrt(dfdata['X1']**2 + dfdata['Y1']**2 + dfdata['Z1']**2)
    leng2 = np.sqrt(dfdata['X2']**2 + dfdata['Y2']**2 + dfdata['Z2']**2)
    leng3 = np.sqrt(dfdata['X3']**2 + dfdata['Y3']**2 + dfdata['Z3']**2)
    vecData = pd.DataFrame({'Time': dfdata['Time'],
                           'RawVectorLen1': np.sqrt((dfdata['X1'].rolling(window=n).mean())**2 + (dfdata['Y1'].rolling(window=n).mean())**2 + (dfdata['Z1'].rolling(window=n).mean())**2),
                           'RawVectorLen2': np.sqrt((dfdata['X2'].rolling(window=n).mean())**2 + (dfdata['Y2'].rolling(window=n).mean())**2 + (dfdata['Z2'].rolling(window=n).mean())**2),
                           'RawVectorLen3': np.sqrt((dfdata['X3'].rolling(window=n).mean())**2 + (dfdata['Y3'].rolling(window=n).mean())**2 + (dfdata['Z3'].rolling(window=n).mean())**2),
                           'Vector3To1': np.sqrt(leng3**2 + leng1**2 - 2 * leng3 * leng1 * cossin(dfdata)),
                           'Vector3To2': np.sqrt(leng3**2 + leng2**2 - 2 * leng3 * leng2 * cossin(dfdata)),
                           'Vector1To2': np.sqrt(leng1**2 + leng2**2 - 2 * leng2 * leng1 * cossin(dfdata))})
    coordDeltaTemp = pd.DataFrame({})
    for sid in ['1', '2', '3']:
        coordDeltaTemp['X' + sid] = dfdata['X' + sid].diff()
        coordDeltaTemp['Y' + sid] = dfdata['Y' + sid].diff()
        coordDeltaTemp['Z' + sid] = dfdata['Z' + sid].diff()
    for sid in ['1', '2', '3']:
        vecData['VectorLenDelta' + sid] = np.sqrt(coordDeltaTemp['X' + sid]**2 + coordDeltaTemp['Y' + sid]**2 + coordDeltaTemp['Z' + sid]**2)
    