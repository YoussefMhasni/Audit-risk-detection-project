import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.experimental import enable_iterative_imputer
import streamlit as st
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
my_form = st.form(key = "form1")
PARA_A = st.number_input('PARA_A')
Risk_A = st.number_input('Risk_A')
Score_B = st.number_input('Score_B');
Score_MV= st.number_input('Score_MV');
District_loss = st.number_input('District loss');
Risk_E = st.number_input('Risk_E');
Score = st.number_input('Score');
Control_Risk = st.number_input('Control_Risk');
submitted = st.button('Submit')



if submitted:
      data=pd.DataFrame(columns=['PARA_A','Risk_A','Score_B','Score_MV','District loss','Risk_E','Score','Control_Risk'])
      data=data.append({'PARA_A':PARA_A,
                      'Risk_A':Risk_A,
                      'Score_B':Score_B,
                      'Score_MV':Score_MV,
                      'District loss':District_loss,
                      'Risk_E':Risk_E,
                      'Score':Score,
                      'Control_Risk':Control_Risk}, ignore_index=True)
      NX_test=data
      st.write(NX_test)
      clf=RandomForestClassifier()
      clf.fit(X_train,y_train)
      st.write("Le résulat de detection est : ")
      st.write(clf.predict(NX_test))


