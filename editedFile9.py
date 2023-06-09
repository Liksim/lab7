import pandas as panda               #commit b
import matplotlib.pyplot as pyplot    #commit b
import numpy as num                #commit b
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


print('Amount of males:',data[data['type']=='M']['humanID'].nunique()) #commit d
print('Amount of females:',data[data['type']=='F']['humanID'].nunique()) #commit d
print('Amount of children:',data[data['type']=='C']['humanID'].nunique()) #commit d


corona = data[data['isNotCoronavirus'] == 'No'] #commit f
print("People who have not had coronavirus:",corona['humanID'].nunique()) #commit f


lungInjury = data[data['isNotLungInjury'] == 'No'] #commit f
print("People without lung damage:",lungInjury['humanID'].nunique()) #commit f


#Функция создает таблицу с вычесленными векторами и их изменением
def makeVectorTableOther(dfdata):
    n = 10 #Для скользящего среднего
    leng1 = num.sqrt(dfdata['X2']**2 + dfdata['Y2']**2 + dfdata['Z2']**2) #commit i
    leng2 = num.sqrt(dfdata['X3']**2 + dfdata['Y3']**2 + dfdata['Z3']**2) #commit i
    leng3 = num.sqrt(dfdata['X1']**2 + dfdata['Y1']**2 + dfdata['Z1']**2) #commit i
    vecData = panda.DataFrame({'Timout': dfdata['Timout'], #commit j
    'RawVectorLen1': num.sqrt((dfdata['X2'].rolling(window=n).mean())**2 + (dfdata['Y2'].rolling(window=n).mean())**2 + (dfdata['Z2'].rolling(window=n).mean())**2), #commit j
    'RawVectorLen2': num.sqrt((dfdata['X3'].rolling(window=n).mean())**2 + (dfdata['Y3'].rolling(window=n).mean())**2 + (dfdata['Z3'].rolling(window=n).mean())**2), #commit j
    'RawVectorLen3': num.sqrt((dfdata['X1'].rolling(window=n).mean())**2 + (dfdata['Y1'].rolling(window=n).mean())**2 + (dfdata['Z1'].rolling(window=n).mean())**2), #commit j
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
    