'''
Created on Sep 23, 2017

@author: Azda Firmansyah
'''
from ExerciseHeartDisease.DataHelper import datasetHeartDisease
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

data=datasetHeartDisease()
X_features = data[0]
Y_labels = data[1]

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_labels, test_size=0.1, random_state=100)
#print(Y_test)
#print("Test Data : ",X_test)
#X_test=[[48,1,2,110,229,0,0,168,0,1,3,0,7]]

print("===Decision Tree Classifier===")
tree_clf = DecisionTreeClassifier(min_samples_split=60)
tree_clf.fit(X_train, Y_train)
pred_tree = tree_clf.predict(X_test)
print("Predict Tree  :",pred_tree)
score_tree = accuracy_score(Y_test, pred_tree)
print("Accuracy Tree :",score_tree)

print("\n===Adaboost Classifier===")
adaboost_clf = AdaBoostClassifier(n_estimators=100)
adaboost_clf.fit(X_train, Y_train)
pred_adaboost = adaboost_clf.predict(X_test)
print("Predict Adaboost :", pred_adaboost)
score_adaboost = accuracy_score(Y_test, pred_adaboost)
print("Accuracy Adaboost:", score_adaboost)

print("\n===Gaussian===")
gauss_clf = GaussianNB()
gauss_clf.fit(X_train, Y_train)
pred_gauss = gauss_clf.predict(X_test)
print("Predict Gaussian :",pred_gauss)
score_gauss = accuracy_score(Y_test, pred_gauss)
print("Accuracy Gaussian:", score_gauss)

print("\n===K Nearest Neighbors")
knneig = KNeighborsClassifier(n_neighbors=20)
knneig.fit(X_train, Y_train)
pred_knneigh = knneig.predict(X_test)
print("Predict Neighbors:",pred_knneigh)
score_knneigh = accuracy_score(Y_test, pred_knneigh)
print("Score KNeighnors :",score_knneigh)

print("\n===Random Forest===")
randfor_clf = RandomForestClassifier(min_samples_split=30)
randfor_clf.fit(X_train, Y_train)
pred_randfor = randfor_clf.predict(X_test)
print("Predict Random Forest :",pred_randfor)
score_randfor = accuracy_score(Y_test, pred_randfor)
print("Score Random Forest :",score_randfor)
"""
Attribute Information:
      -- 1. (age)       
      -- 2. #4  (sex)       
      -- 3. #9  (cp)        
      -- 4. #10 (trestbps)  
      -- 5. #12 (chol)      
      -- 6. #16 (fbs)       
      -- 7. #19 (restecg)   
      -- 8. #32 (thalach)   
      -- 9. #38 (exang)     
      -- 10. #40 (oldpeak)   
      -- 11. #41 (slope)     
      -- 12. #44 (ca)        
      -- 13. #51 (thal)
      More information visit : https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names

#X_test=[[48,1,2,110,229,0,0,168,0,1,3,0,7]]
#X_test=[[48,1,2,110,229,0,0,168,0,1,3,0,7]]
features_train, labels_train, features_test, labels_test = train_test_split(X,Y, test_size=0.3, random_state=42)

#Tree Clf
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X, Y)
tree_predict = tree_clf.predict(features_test)
print(tree_predict)
"""
#tree_score = accuracy_score(labels_test, tree_predict)
#print(tree_score)