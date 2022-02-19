import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.experimental import enable_iterative_imputer

from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('C:/Users/youss/Desktop/S3/SI/audit_data.csv')
df= pd.read_csv('C:/Users/youss/Desktop/S3/SI/audit_data.csv')

import io
def info(df):
    buffer = io.StringIO()
    df.info(verbose=True,buf=buffer,null_counts=None)
    s = buffer.getvalue()
    st.text(s)
    return

def datainfo():
    st.header('Présentation du jeu de données')
    st.header('Jeu de données')
    st.write(df)
    st.header('Description')
    st.write(df.describe())
    st.header('Informations du jeu de données')
    info(df)

def visualisation():
    st.header('Visualisation du jeu de données')
    st.header('Histogrammes')
    df.hist(figsize=(16, 20))
    plt.show()
    st.pyplot()
    st.subheader('Boite à moustaches')
    sns.set(rc={'figure.figsize': (30, 10)})
    sns.boxplot(data=df.select_dtypes(include='number'))
    st.pyplot()
    st.subheader('Matrice de corrélation')
    f, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(f)
def datacleaning():
    st.header("Nettoyage des données")
    info(df)
    st.header('Gestion des données manquantes')
    st.write(df.isnull().sum())
    st.write("pour faire face au probleme de données manquantes dans une ligne de la colonne 'Money_value' , on va utilise un algorithme s'appelle IterativeImputer pour predir la case manquantes")
    imp = IterativeImputer(max_iter=10, random_state=0)
    dat=pd.DataFrame(imp.fit_transform(df))
    st.write(dat.isnull().sum())
    st.subheader("Standarisation")
    MMS=MinMaxScaler()
    b=MMS.fit_transform(dat)
    data_norm=pd.DataFrame(b)
    st.write(data_norm.head(10))
def cleaning():
    df.drop('Detection_Risk',axis=1,inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop([351, 355, 367],0,inplace=True)
    df["LOCATION_ID"] = pd.to_numeric(df["LOCATION_ID"])
cleaning()
imp = IterativeImputer(max_iter=10, random_state=0)
data=pd.DataFrame(imp.fit_transform(df))

MMS=MinMaxScaler()
b=MMS.fit_transform(data)
data_norm=pd.DataFrame(b)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(0.5))
sel.fit_transform(data)
y=data_norm[25]
X=data_norm.drop(25,axis=1)
X1=data.drop(25,axis=1)
selector=SelectFromModel(SGDClassifier(random_state=0),threshold='mean')
X_selecte=pd.DataFrame(selector.fit_transform(X,y))
X_train,X_test,y_train,y_test=train_test_split(X_selecte,y)


def LogisticRgression():
    clf=LogisticRegression()
    clf.fit(X_train,y_train)
    st.write("L'algorithme Logistic Regression donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return
def KNearest():
    clf=KNeighborsClassifier(n_neighbors=2)
    clf.fit(X_train,y_train)
    st.write("L'algorithme KNeighborsClassifier donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return

def DecisionTree():
    clf=DecisionTreeClassifier(random_state=0)
    clf.fit(X_train,y_train)
    st.write("L'algorithme Decision Tree donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return
def LineSVC():
    clf=LinearSVC()
    clf.fit(X_train,y_train)
    st.write("L'algorithme Support Vector Machine (Linear Kernel) donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return
def  svc():
    clf=SVC()
    clf.fit(X_train,y_train)
    st.write("L'algorithme Support Vector Machine (RBF Kernel) donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return
def  MLP():
    clf=MLPClassifier()
    clf.fit(X_train,y_train)
    st.write("L'algorithme Neural Network donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return
def clf():
    clf=RandomForestClassifier()
    clf.fit(X_train,y_train)
    st.write("L'algorithme CLF donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return
def GradientClassifier():
    clf=GradientBoostingClassifier()
    clf.fit(X_train,y_train)
    st.write("L'algorithme Gradient Boosting donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return
def BaggingClassifie():
    clf=BaggingClassifier(base_estimator=KNeighborsClassifier(),n_estimators=100)
    clf.fit(X_train,y_train)
    st.write("L'algorithme Bagging donne le résultat avec une précision de : ")
    st.write(clf.score(X_train,y_train))
    return


def Algos():
    st.sidebar.header('Algorithmes de classification')
    LogisticRegression=st.sidebar.button('Logistic Regression',key=1, help=None, on_click=LogisticRgression, args=None, kwargs=None)
    KNeighborsClassifier=st.button('K-Nearest Neighbors',key=2, help=None, on_click=KNearest, args=None, kwargs=None)
    DecisionTreeClassifier=st.button('Decision Tree',key=3, help=None, on_click=DecisionTree, args=None, kwargs=None)
    LinearSVC=st.button('Support Vector Machine (Linear Kernel)',key=4, help=None, on_click=LineSVC, args=None, kwargs=None)
    SVC=st.button('Support Vector Machine (RBF Kernel)',key=5, help=None, on_click=svc, args=None, kwargs=None)
    Neural_Network=st.button('Neural Network', key=6,help=None, on_click=MLP, args=None, kwargs=None)
    RandomForestClassifier=st.button('Random Forest',key=7, help=None, on_click=clf, args=None, kwargs=None)
    GradientBoostingClassifier=st.button('Gradient Boosting',key=8, help=None, on_click=GradientClassifier, args=None, kwargs=None)
    BaggingClassifier=st.button('Bagging', key=9,help=None, on_click=BaggingClassifie, args=None, kwargs=None)

def get_classifier(clf_name):
    clf = None
    if clf_name == 'Logistic Regression':
        clf = LogisticRgression()
    elif clf_name == 'K-Nearest Neighbors':
        clf = KNearest()
    elif clf_name == 'Decision Tree':
        clf = DecisionTree()
    elif clf_name == 'Support Vector Machine (Linear Kernel)':
        clf =LineSVC()
    elif clf_name == 'Support Vector Machine (RBF Kernel)':
        clf = svc()
    elif clf_name == 'Neural Network':
        clf = MLP()
    elif clf_name == 'Random Forest':
        clf = clf()
    elif clf_name == 'Gradient Boosting':
        clf =GradientClassifier()
    elif clf_name == 'Bagging':
        clf = BaggingClassifie()
    return clf
def get_data():
    return []
def Prediction():
    user_id = st.text_input("User ID")
    foo = st.slider("foo", 0, 100)
    bar = st.slider("bar", 0, 100)

    if st.button("Add row"):
        get_data().append({"UserID": user_id, "foo": foo, "bar": bar})

    st.write(pd.DataFrame(get_data()))
    return
def user_input():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    Data=st.sidebar.button('Jeu de données',key=1, help=None, on_click=datainfo, args=None, kwargs=None)
    Visualisation=st.sidebar.button('Visualisation',key=1, help=None, on_click=visualisation, args=None, kwargs=None)
    Datacleaning=st.sidebar.button('Pré-traitement des données',key=1, help=None, on_click=datacleaning, args=None, kwargs=None)
    clf_name=st.sidebar.selectbox('Algorithmes de classification',
    ('','Logistic Regression', 'K-Nearest Neighbors','Decision Tree','Support Vector Machine (Linear Kernel)',
     'Support Vector Machine (RBF Kernel)', 'Neural Network','Random Forest', 'Gradient Boosting', 'Bagging'))
    get_classifier(clf_name)
    dataf=pd.DataFrame(df)

    pred=st.sidebar.button('Prédiction',key=1, help=None,on_click=Prediction)

    return dataf
df=user_input()
