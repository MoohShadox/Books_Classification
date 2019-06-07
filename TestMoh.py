from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn import preprocessing
import re
import numpy as np
from sklearn.feature_selection import SelectKBest , chi2 , f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold , StratifiedKFold , GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from Gestion_Datasets import Benchmarking_Datasets

def afficherNombreElemParClasse(target_a):
    nb_classes = {}
    target = list (target_a)
    sum = 0
    for i in target:
        nb_classes[i] = nb_classes.get (i , 0) + 1
        sum = sum + 1
    for i in nb_classes:
        print (i , " : " , nb_classes[i])
        nb_classes[i] = nb_classes[i] / sum
    return nb_classes

def traitementClasse(data_a,target_a,classes_a_concerver):
    print("Début du traitement des classes")
    descriptions_classes = {}
    nb_classes = {}
    target = list (target_a)
    data = list(data_a)
    cpt = 0
    for i in target:
        descriptions_classes[i] = descriptions_classes.get (i , []) + [data[cpt]]
        nb_classes[i] = nb_classes.get(i,0) + 1
        cpt = cpt + 1

    print("fin du parcours")



    print("fin du tri")
    ndata = []
    ntarget = []
    for i in range(0,data_a.shape[0]):
        if(target[i] in classes_a_concerver):
            ndata.append(data_a[i])
            ntarget.append(target[i])
    ndata = np.array(ndata)
    ntarget = np.array(ntarget)
    print("Data shape = ",ndata.shape)
    print("Target shape = ",ntarget.shape)
    return ndata,ntarget

def benchMarkNawel(data,target,stratified = True):
    print("debut benchmarking")
    if(stratified):
        kf = StratifiedKFold (n_splits=10)
        LDA = LinearDiscriminantAnalysis()
        kf.get_n_splits (data)
        print ("nawel")
        vect = []
        for train_index , test_index in kf.split (data,target):
            X_train , X_test = data[train_index] , data[test_index]
            y_train , y_test = target[train_index] , target[test_index]
            LDA.fit(X_train,y_train)
            X_train = LDA.transform(X_train)
            X_test = LDA.transform(X_test)
            #clf = svm.SVC(kernel = "rbf",gamma='scale', decision_function_shape='ova').fit (X_train , y_train) # 77.7 %
            #clf = RandomForestClassifier(n_estimators=200).fit (X_train , y_train) #75 %
            #clf = LogisticRegression ().fit (X_train , y_train) #75
            #clf = MLPClassifier (hidden_layer_sizes=(500,),max_iter=1500).fit (X_train , y_train) #75% la bonne surinterprétation entre 23 et 91%
            #clf = svm.LinearSVC (multi_class='ovr',class_weight=afficherNombreElemParClasse(y_train)).fit (X_train , y_train) #75 %
            prediction = clf.predict ( X_test)
            score = precision_score(y_test,prediction,average="weighted")
            print ("score :" + str (score))
            vect.append (score)
    else:
        kf = KFold (n_splits=3)
        kf.get_n_splits (data)
        print ("nawel")
        vect = []
        for train_index , test_index in kf.split (data ):
            X_train , X_test = data[train_index] , data[test_index]
            y_train , y_test = target[train_index] , target[test_index]
            clf = RandomForestClassifier (n_estimators=50).fit (X_train , y_train)
            prediction = clf.predict (X_test)
            score = precision_score (y_test , prediction , average="weighted")
            print ("score :" + str (score))
            vect.append (score)
    import numpy as np
    print ("Moyenne des scores de précisions : " , np.array (vect).mean ())
    return np.array (vect).mean ()


def benchMarkGridSearch(data,target):
    grid = {'kernel': ('linear' , 'rbf') , 'C': [1 , 10, 100, 1000] , 'gamma': [0.001, 0.0001, 0.01]}
    clf = svm.SVC ( decision_function_shape='ova',gamma="scale" )
    GSCV = GridSearchCV (clf , param_grid=grid,cv=10)
    kf = StratifiedKFold (n_splits=5)
    LDA = LinearDiscriminantAnalysis ()
    kf.get_n_splits (data)
    print ("nawel")
    vect = []
    for train_index , test_index in kf.split (data , target):
        X_train , X_test = data[train_index] , data[test_index]
        y_train , y_test = target[train_index] , target[test_index]
        LDA.fit (X_train , y_train)
        X_train = LDA.transform (X_train)
        X_test = LDA.transform (X_test)
        GSCV.fit(X_train,y_train)
        prediction = GSCV.predict (X_test)
        score = precision_score (y_test , prediction , average="weighted")
        print ("score :" + str (score))
        vect.append (score)
    import numpy as np
    print ("Moyenne des scores de précisions : " , np.array (vect).mean ())



def writeProjection(data):
    P = PCA (n_components=500)
    data = P.fit_transform (data)
    import pandas as pd
    df = pd.DataFrame (data)
    f = open ("PCAProjection450.csv" , "w")
    df.to_csv ("PCAProjection450.csv")


def readProjection():
    import pandas as pd
    df = pd.read_csv("PCAProjection450.csv")
    ar = np.array(df)
    return ar

def writeTarget(target):
    import pandas as pd
    df = pd.DataFrame (target)
    f = open ("target.csv" , "w")
    df.to_csv ("target.csv")

def readTarget():
    import pandas as pd
    df = pd.read_csv("target.csv")
    ar = np.array(df)
    return ar

knn = KNeighborsClassifier()
logreg = LogisticRegression()
tfidf_transformer = TfidfTransformer()
le = preprocessing.LabelEncoder()
item = pickle.load(open('Classification', 'rb'))
ListAut=[]
inter=[]
l=[]
ListEdit=[]
o = 0
for i in item:
    o=o+1
    ListAut.append(i[2])
    inter.append(i[4])
    l.append(i[5])
    ListEdit.append(i[3])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(inter)
print(X.shape)
X_train_tfidf = tfidf_transformer.fit_transform(X)
le.fit(l)
target=le.transform(l)
le.fit(ListAut)
y=le.transform(ListAut)
le.fit(ListEdit)
j=le.transform(ListEdit)
data=np.concatenate((X_train_tfidf.toarray(),y[:,None]),axis=1)
print("Data = ",data.shape)
data,target = traitementClasse(data,target,[2,1,5,6,7])
#writeProjection(data)
#writeTarget(target)
data = readProjection()
#target = readTarget()
#data = SelectKBest(f_classif, k=500).fit_transform(data, target)
#writeProjection(data)
print("Data = ",data.shape)
#Benchmarking_Datasets_Nawel.evaluateDataTarget(data,target)
print("Benchmarking Nawel : ")
nb_points = 3
pas = data
rez = []
for i in range(100,data.shape[0],int(data.shape[0] / nb_points)):
    data_c = data[:i]
    target_c = target[:i]
    rez.append([benchMarkNawel(data_c,target_c) , i])

arr = np.array(rez)
import pandas as pd
df = pd.DataFrame(arr)
print(df)
import seaborn as sns
sns.set ()
import matplotlib.pyplot as plt
ax = sns.lineplot (x="0" , y="1" ,  data=rez)
plt.show ()