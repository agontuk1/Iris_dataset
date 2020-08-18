import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('iris.csv')
X=dataset.iloc[:, [0,1,2,3]].values
y=dataset.iloc[:, 4].values
              
g = sns.pairplot(dataset, hue='species', markers='+')
plt.show()
              
g = sns.violinplot(y='species', x='sepal_length', data=dataset, inner='quartile')
plt.show()
g = sns.violinplot(y='species', x='sepal_width', data=dataset, inner='quartile')
plt.show()
g = sns.violinplot(y='species', x='petal_length', data=dataset, inner='quartile')
plt.show()
g = sns.violinplot(y='species', x='petal_width', data=dataset, inner='quartile')
plt.show()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)
logreg.fit(X_train,y_train)

y_predlogreg=logreg.predict(X_test)

accuracylogreg=accuracy_score(y_true=y_test,y_pred=y_predlogreg)*100
                       
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train, y_train)

y_predsvm=svm.predict(X_test)

accuracysvm=accuracy_score(y_true=y_test,y_pred=y_predsvm)*100
                       
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=12, p=2, metric='minkowski')
knn.fit(X_train, y_train)

y_predknn=knn.predict(X_test)

accuracyknn=accuracy_score(y_true=y_test,y_pred=y_predknn)*100
                       
import xgboost as xgb

xgb = xgb.XGBClassifier()
xgb = xgb.fit(X_train, y_train)

y_predxgb=logreg.predict(X_test)

accuracyxgb=accuracy_score(y_true=y_test,y_pred=y_predxgb)*100