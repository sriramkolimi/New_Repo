#future prediction for car purchases datasetimport numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv(r"C:\Users\Dell\Downloads\DS\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#******************************************************************************
#logistic regression
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
logistic_regression_pred = logistic_regression.predict(X_test)


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,logistic_regression_pred)
cm1

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_test,logistic_regression_pred)
ac1

#******************************************************************************
#Random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train)
random_forest_pred = random_forest.predict(X_test)

cm2=confusion_matrix(y_test,random_forest_pred)
cm2

ac2 = accuracy_score(y_test,random_forest_pred)
ac2

#******************************************************************************
#KNN

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7,p=2,weights='uniform',algorithm='auto')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

cm3 = confusion_matrix(y_test, knn_pred)
cm3

ac3 = accuracy_score(y_test, knn_pred)
ac3

#*****************************************************************************
#SVM
from sklearn.svm import SVC
svm= SVC( )
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

cm4 = confusion_matrix(y_test, svm_pred)
cm4

ac4= accuracy_score(y_test, svm_pred)
ac4

#******************************************************************************
#decision tree

from sklearn.tree import DecisionTreeClassifier
decision_tree= DecisionTreeClassifier()
decision_tree.fit(X_train, y_train) 
decision_tree_pred = decision_tree.predict(X_test)
cm5 = confusion_matrix(y_test, decision_tree_pred)
cm5
ac5= accuracy_score(y_test, decision_tree_pred)
ac5
#******************************************************************************
dataset1 = pd.read_csv(r"C:\Users\Dell\Downloads\DS\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Future prediction1.csv")
d2 = dataset1.copy()
dataset1 = dataset1.iloc[:, [2, 3]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)
#*****************************************************************************
logistic_regression_pred1= pd.DataFrame()
d2 ['logistic_regression_pred1'] = logistic_regression.predict(M)
d2.to_csv('final1.csv')
#******************************************************************************
random_forest_pred1= pd.DataFrame()
d2 ['random_forest_pred1'] = random_forest.predict(M)
d2.to_csv('final1.csv')
#******************************************************************************
knn_pred1= pd.DataFrame()
d2 ['knn_pred1'] = knn.predict(M)
d2.to_csv('final1.csv')
#******************************************************************************
svm_pred1= pd.DataFrame()
d2 ['svm_pred1'] = svm.predict(M)
d2.to_csv('final1.csv')
#******************************************************************************
decision_tree_pred1= pd.DataFrame()
d2 ['decision_tree_pred1'] = decision_tree.predict(M)
d2.to_csv('final1.csv')


#========================================================================================================

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\Data_Science_Notes\May\15. Logistic regression with future prediction\Social_Network_Ads.csv")

dataset.head()

x = dataset.iloc[:, [2,3]].values
print(x)
y = dataset.iloc[:, -1].values
print(y)

#Splitting training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(x_test)
#-----------------------------------------------------------------------------------------------------
# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression()
classifier1.fit(x_train, y_train)

# Predicting a new result
y_pred = classifier1.predict(x_test)
print(y_pred)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
 
# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

#----------------------------------------------------------------------------------------------------
# Training the KNN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier()
classifier2 = classifier2.fit(x_train, y_train)

# Predicting a new result
y_pred1 = classifier2.predict(x_test)
print(y_pred1)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred1)
print(cm_knn)

from sklearn.metrics import accuracy_score
ac_knn = accuracy_score(y_test, y_pred1)
print(ac_knn)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr_knn = classification_report(y_test, y_pred1)
print(cr_knn)

#----------------------------------------------------------------------------------------------------
#Training the SVM model on the Training set
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'rbf', random_state = 0)
classifier3.fit(x_train, y_train)

# Predicting a new result
y_pred1 = classifier3.predict(x_test)
print(y_pred1)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred1)
print(cm_svm)

from sklearn.metrics import accuracy_score
ac_svm = accuracy_score(y_test, y_pred1)
print(ac_svm)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr_svm = classification_report(y_test, y_pred1)
print(cr_svm)

#----------------------------------------------------------------------------------------------------
#Future prediction analysis
dataset1 = pd.read_csv(r"C:\Users\Admin\Desktop\Data_Science_Notes\May\15. Logistic regression with future prediction\Future prediction1.csv")
d2 = dataset1.copy()
dataset1 = dataset1.iloc[:, [2, 3]].values

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

#Prediction results
logistic_pred = pd.DataFrame()
knn_pred = pd.DataFrame()
svm_pred = pd.DataFrame()

d2 ['logistic_pred'] = classifier1.predict(M)
d2 ['knn_pred'] = classifier2.predict(M)
d2 ['svm_pred'] = classifier3.predict(M)
d2.to_csv('Models_Comparision_File.csv')