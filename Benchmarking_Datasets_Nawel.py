import time

from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from sklearn.metrics import precision_score, average_precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate


import numpy as np


from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

scoring_binary = {
    'Recall': make_scorer(recall_score, average='weighted'),
    'Precision': make_scorer(precision_score, average='weighted'),
    'F1Score': make_scorer(f1_score, average='weighted'),
    'Accuracy': make_scorer(accuracy_score),
    'ROCAUC': make_scorer(recall_score, average='weighted'),
    "True Positif" : make_scorer(tp),
    "False Positif": make_scorer(fp),
    "False Negatif": make_scorer(fn),
    "True Negatf" : make_scorer(tn),
}

scoring = {
    'Recall': make_scorer(recall_score, average='weighted'),
    'Precision': make_scorer(precision_score, average='weighted'),
    'F1Score': make_scorer(f1_score, average='weighted'),
    'Accuracy': make_scorer(accuracy_score),
    'ROCAUC': make_scorer(recall_score, average='weighted'),
}

scoring_nawel = {
    'Recall': make_scorer(recall_score, average='weighted'),
    'Accuracy': make_scorer(accuracy_score),
    'ROCAUC': make_scorer(recall_score, average='weighted'),
}


modele_map = {}

#liste_modeles = ["BN","RF","LSVM","RBFSVM","GaussianProcess","AdaBoost","QDA","KNN","DTC","MLP"]

liste_type_classeurs = ["Arbre","Bayesiens","Linaire","Perceptron"]

liste_modeles = ["BN","RF","GaussianProcess","AdaBoost","QDA","KNN","DTC","MLP"]

liste_modeles_nawel = ["BN","RF","AdaBoost","KNN","DTC","MLP"]

noms_classeurs = {
    "BN" : "Classeur Naif de Bayes",
    "RF" : "Classeur par forêt aléatoire"
}

type_classeurs = {
    'BN' : 1,
    "RF" : 0,
    "RBFSVM" : 2
}

resultats_benchmark = {}





def modele_generator(motclef):
    if (motclef == "BN"):
        return GaussianNB ()
    elif (motclef == "DTC"):
        return DecisionTreeClassifier ()
    elif (motclef == "LSVM"):
        return LinearSVC ()
    elif (motclef == "RBFSVM"):
        return SVC (kernel='rbf')
    elif (motclef == "GaussianProcess"):
        return GaussianProcessClassifier ()
    elif (motclef == "AdaBoost"):
        return AdaBoostClassifier ()
    elif (motclef == "QDA"):
        return QuadraticDiscriminantAnalysis ()
    elif (motclef == "KNN"):
        return KNeighborsClassifier ()
    elif (motclef == "RF"):
        return RandomForestClassifier (n_estimators=50)
    elif (motclef == 'MLP'):
        return MLPClassifier ()
    elif (motclef == "MNB"):
        return MultinomialNB()



def prepareModels():
    for i in liste_modeles:
        modele_map[i] = modele_generator(i)

def evaluateDataTarget(data,target,nfolds = 3):
    Rez = {}
    for model in liste_modeles_nawel:
        m = modele_generator(model)
        print ("Modèle = " , model)
        D = cross_validate (m , data , target , scoring=scoring_nawel , cv=nfolds , return_train_score=False)
        for i in D:
            print(i.replace("test_","")," : ",np.array(D[i]).mean())
        Rez[model] = D
    return Rez



#repareModels()
#evaluateModel("Australian","RF")

