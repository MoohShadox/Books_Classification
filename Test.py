from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn import preprocessing
import re
import numpy as np
from sklearn.feature_selection import SelectKBest , chi2 , f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
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
    for i in target:
        nb_classes[i] = nb_classes.get (i , 0) + 1
    for i in nb_classes:
        print (i , " : " , nb_classes[i])

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
    if(stratified):
        kf = StratifiedKFold (n_splits=3)
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
            clf = RandomForestClassifier (n_estimators=50).fit (X_train , y_train)
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

#data = SelectKBest(f_classif, k=500).fit_transform(data, target)
P = PCA(n_components= 450)
data = P.fit_transform(data)
print("Data = ",data.shape)
#Benchmarking_Datasets_Nawel.evaluateDataTarget(data,target)
print("Benchmarking Nawel : ")
benchMarkNawel(data,target)

