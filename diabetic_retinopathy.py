# -*- coding: utf-8 -*-
"""
@author: Phongphat Wiwatthanasetthakarn
"""

import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import Experiment as exp

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score


data = arff.loadarff(r"./data_file/messidor_features.arff")
df = pd.DataFrame(data[0])


"""Check null value in every single feature"""
pd.isnull(df).sum()


"""Mapping Class Variable"""
df["Class"] = df["Class"].map({b"0": 0, b"1": 1})

column_names =  {
                      "0": "qa"
                    , "1": "pre_screening"
                    
                    , "2": "ma_05"
                    , "3": "ma_06"
                    , "4": "ma_07"
                    , "5": "ma_08"
                    , "6": "ma_09"
                    , "7": "ma_10"
                    
                    , "8": "ex_01"
                    , "9": "ex_02"
                    , "10": "ex_03"
                    , "11": "ex_04"
                    , "12": "ex_05"
                    , "13": "ex_06"
                    , "14": "ex_07"   
                    , "15": "ex_08"
                    
                    , "16": "distance"
                    , "17": "optic_disc_dia"
                    , "18": "am_fm"
                }
               
df = df.rename(columns=column_names)
print ("\n df.head()")
print ( df.head() )

"""
======================================
Check relation between feature and target to reduce similar features
======================================
"""
df_relation = df.corr(method='pearson')
print ("\n df_relation")
print ( df_relation )


"""ma_05 = 0.292603 is highest Pearson correlation coefficient """
"""ex_07 = 0.184772 is highest Pearson correlation coefficient"""

X_features = df[ ["qa", "pre_screening", "ma_05", "ex_07", "distance", "optic_disc_dia", "am_fm" ] ]
y_features = df[ ["Class"] ]

print ("\n X_features")
print ( X_features )

print ("\n y_features")
print ( y_features )


"""Check Missing Value"""
print ( "\n y_features['Class'].value_counts()" )
print ( y_features['Class'].value_counts() )


"""
======================================
Fitting model_00: Logistic Regression (baseline)
Include all reducing features without ranking the important of features 
======================================
"""

"""Sampling a training set and a testing set = 80:20"""
X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(X_features, y_features, test_size = 0.20, random_state=0, stratify=y_features)


model_logr_baseline = exp.Experiment(X = X_train_baseline, y = y_train_baseline).logistic_regression()

"""Intercept and Coefficients"""

print( "\n Intercept \n" + str(model_logr_baseline["tuned_model"].intercept_[0]) )
print( "\n Coefficients \n" + str(model_logr_baseline["tuned_model"].coef_[0]) )

"""Evaluate and predict"""

model_logr_baseline["pop_DR"] = model_logr_baseline["tuned_model"].predict_proba(X_test_baseline)[:, 1]
model_logr_baseline["yhat"] = model_logr_baseline["tuned_model"].predict(X_test_baseline)

model_logr_baseline["con_matrix"] = confusion_matrix(y_test_baseline, model_logr_baseline["yhat"], labels=[0, 1])
model_logr_baseline["class_report"] = classification_report(y_test_baseline, model_logr_baseline["yhat"])

model_logr_baseline["roc"] = roc_curve(y_test_baseline, model_logr_baseline["pop_DR"], pos_label=1)
model_logr_baseline["auc"] = round(roc_auc_score(y_test_baseline, model_logr_baseline["pop_DR"]), 2)

print( "\n pop_DR" )
print( model_logr_baseline["pop_DR"] )

print( "\n yhat" )
print( model_logr_baseline["yhat"] )

print( "\n auc" )
print( model_logr_baseline["auc"] )


"""Confusion Matrix"""

print( """\n model_logr_baseline["con_matrix"]""" )
print( model_logr_baseline["con_matrix"] )

# True Positives
model_logr_baseline["TP"] = model_logr_baseline["con_matrix"][1, 1]

# True Negatives
model_logr_baseline["TN"] = model_logr_baseline["con_matrix"][0, 0]

# False Positives
model_logr_baseline["FP"] = model_logr_baseline["con_matrix"][0, 1]

# False Negatives
model_logr_baseline["FN"] = model_logr_baseline["con_matrix"][1, 0]

model_logr_baseline["accuracy"] = (model_logr_baseline["TP"] + model_logr_baseline["TN"]) / float(model_logr_baseline["TP"] + model_logr_baseline["TN"] + model_logr_baseline["FP"] + model_logr_baseline["FN"])
model_logr_baseline["recall"] = model_logr_baseline["TP"] / float(model_logr_baseline["TP"] + model_logr_baseline["FN"])
model_logr_baseline["specificity"] = model_logr_baseline["TN"] / float(model_logr_baseline["TN"] + model_logr_baseline["FP"])
model_logr_baseline["precision"] = model_logr_baseline["TP"] / float(model_logr_baseline["TP"] + model_logr_baseline["FP"])

print("\n Model performance calculate from confusion matrix:\n" )
print("accuracy: ", model_logr_baseline["accuracy"] )
print("recall: ",model_logr_baseline["recall"] )
print("specificity: ",model_logr_baseline["specificity"] )
print("precision: ",model_logr_baseline["precision"] )

print ( """\n model_logr_baseline["class_report"]""" )
print ( model_logr_baseline["class_report"] )


"""
======================================
Feature selection by ranking the important of features
======================================
"""

"""fitting the model to select important features"""
clf_ex_tree = ExtraTreesClassifier()
clf_ex_tree.fit(X_features, y_features)


"""show importance features """
print( "\n clf_ex_tree.feature_importances_")
print( clf_ex_tree.feature_importances_ )


"""sort importance features"""
feature_imp_argsort = clf_ex_tree.feature_importances_.argsort()
feature_top_most_from_last = [-1,-2,-3,-4,-5]
feature_imp_argsort_top_most = feature_imp_argsort[feature_top_most_from_last]

print( "\n X_features.columns[feature_imp_argsort_top_most]")
print( X_features.columns[feature_imp_argsort_top_most] )

print( "\n clf_ex_tree.feature_importances_[feature_imp_argsort_top_most]")
print( clf_ex_tree.feature_importances_[feature_imp_argsort_top_most] )

df_feature_imp = pd.DataFrame(
                                {
                                    'features':X_features.columns[feature_imp_argsort_top_most],
                                    'important':clf_ex_tree.feature_importances_[feature_imp_argsort_top_most]
                                }
                            )

ax = df_feature_imp.plot.bar(x='features', y='important', rot=85)


"""
======================================
Dividing the data into training and testing sets
======================================
"""
X = X_features[ X_features.columns[feature_imp_argsort_top_most] ]
y = y_features

"""Sampling a training set and a testing set = 80:20"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0, stratify=y)


print ("\nX_train.shape")
print ( X_train.shape )

print ("\nX_test.shape")
print ( X_test.shape )

print ("\ny_train.shape")
print ( y_train.shape )

print ("\ny_test.shape")
print ( y_test.shape )


"""
======================================
Fitting model_01: Logistic Regression
======================================
"""
model_logr = exp.Experiment(X = X_train, y = y_train).logistic_regression()

"""Intercept and Coefficients"""

print( "\n Intercept \n" + str(model_logr["tuned_model"].intercept_[0]) )
print( "\n Coefficients \n" + str(model_logr["tuned_model"].coef_[0]) )

"""Evaluate and predict"""

model_logr["pop_DR"] = model_logr["tuned_model"].predict_proba(X_test)[:, 1]
model_logr["yhat"] = model_logr["tuned_model"].predict(X_test)

model_logr["con_matrix"] = confusion_matrix(y_test, model_logr["yhat"], labels=[0, 1])
model_logr["class_report"] = classification_report(y_test, model_logr["yhat"])

model_logr["roc"] = roc_curve(y_test, model_logr["pop_DR"], pos_label=1)
model_logr["auc"] = round(roc_auc_score(y_test, model_logr["pop_DR"]), 2)

print( "\n pop_DR" )
print( model_logr["pop_DR"] )

print( "\n yhat" )
print( model_logr["yhat"] )

print( "\n auc" )
print( model_logr["auc"] )


"""Confusion Matrix"""

print( """\n model_logr["con_matrix"]""" )
print( model_logr["con_matrix"] )

# True Positives
model_logr["TP"] = model_logr["con_matrix"][1, 1]

# True Negatives
model_logr["TN"] = model_logr["con_matrix"][0, 0]

# False Positives
model_logr["FP"] = model_logr["con_matrix"][0, 1]

# False Negatives
model_logr["FN"] = model_logr["con_matrix"][1, 0]

model_logr["accuracy"] = (model_logr["TP"] + model_logr["TN"]) / float(model_logr["TP"] + model_logr["TN"] + model_logr["FP"] + model_logr["FN"])
model_logr["recall"] = model_logr["TP"] / float(model_logr["TP"] + model_logr["FN"])
model_logr["specificity"] = model_logr["TN"] / float(model_logr["TN"] + model_logr["FP"])
model_logr["precision"] = model_logr["TP"] / float(model_logr["TP"] + model_logr["FP"])

print("\n Model performance calculate from confusion matrix:\n" )
print("accuracy: ", model_logr["accuracy"] )
print("recall: ",model_logr["recall"] )
print("specificity: ",model_logr["specificity"] )
print("precision: ",model_logr["precision"] )

print ( """\n model_logr["class_report"]""" )
print ( model_logr["class_report"] )


"""
======================================
Fitting model_02: Random forest
======================================
"""
model_rf = exp.Experiment(X = X_train, y = y_train).random_forest()

"""Evaluate and predict"""

model_rf["pop_DR"] = model_rf["tuned_model"].predict_proba(X_test)[:, 1]
model_rf["yhat"] = model_rf["tuned_model"].predict(X_test)

model_rf["con_matrix"] = confusion_matrix(y_test, model_rf["yhat"], labels=[0, 1])
model_rf["class_report"] = classification_report(y_test, model_rf["yhat"])

model_rf["roc"] = roc_curve(y_test, model_rf["pop_DR"], pos_label=1)
model_rf["auc"] = round(roc_auc_score(y_test, model_rf["pop_DR"]), 2)

print( "\n pop_DR" )
print( model_rf["pop_DR"] )

print( "\n yhat" )
print( model_rf["yhat"] )

print( "\n auc" )
print( model_rf["auc"] )


"""Confusion Matrix"""

print( """\n model_rf["con_matrix"]""" )
print( model_rf["con_matrix"] )

# True Positives
model_rf["TP"] = model_rf["con_matrix"][1, 1]

# True Negatives
model_rf["TN"] = model_rf["con_matrix"][0, 0]

# False Positives
model_rf["FP"] = model_rf["con_matrix"][0, 1]

# False Negatives
model_rf["FN"] = model_rf["con_matrix"][1, 0]

model_rf["accuracy"] = (model_rf["TP"] + model_rf["TN"]) / float(model_rf["TP"] + model_rf["TN"] + model_rf["FP"] + model_rf["FN"])
model_rf["recall"] = model_rf["TP"] / float(model_rf["TP"] + model_rf["FN"])
model_rf["specificity"] = model_rf["TN"] / float(model_rf["TN"] + model_rf["FP"])
model_rf["precision"] = model_rf["TP"] / float(model_rf["TP"] + model_rf["FP"])

print("\n Model performance calculate from confusion matrix:\n" )
print("accuracy: ", model_rf["accuracy"] )
print("recall: ",model_rf["recall"] )
print("specificity: ",model_rf["specificity"] )
print("precision: ",model_rf["precision"] )

print ( """\n model_rf["class_report"]""" )
print ( model_rf["class_report"] )


"""
======================================
Fitting model_03: Gaussian Naive Bayes
======================================
"""
model_gnb = exp.Experiment(X = X_train, y = y_train).gnb()

"""Evaluate and predict"""

model_gnb["pop_DR"] = model_gnb["tuned_model"].predict_proba(X_test)[:, 1]
model_gnb["yhat"] = model_gnb["tuned_model"].predict(X_test)

model_gnb["con_matrix"] = confusion_matrix(y_test, model_gnb["yhat"], labels=[0, 1])
model_gnb["class_report"] = classification_report(y_test, model_gnb["yhat"])

model_gnb["roc"] = roc_curve(y_test, model_gnb["pop_DR"], pos_label=1)
model_gnb["auc"] = round(roc_auc_score(y_test, model_gnb["pop_DR"]), 2)

print( "\n pop_DR" )
print( model_gnb["pop_DR"] )

print( "\n yhat" )
print( model_gnb["yhat"] )

print( "\n auc" )
print( model_gnb["auc"] )


"""Confusion Matrix"""

print( """\n model_gnb["con_matrix"]""" )
print( model_gnb["con_matrix"] )

# True Positives
model_gnb["TP"] = model_gnb["con_matrix"][1, 1]

# True Negatives
model_gnb["TN"] = model_gnb["con_matrix"][0, 0]

# False Positives
model_gnb["FP"] = model_gnb["con_matrix"][0, 1]

# False Negatives
model_gnb["FN"] = model_gnb["con_matrix"][1, 0]

model_gnb["accuracy"] = (model_gnb["TP"] + model_gnb["TN"]) / float(model_gnb["TP"] + model_gnb["TN"] + model_gnb["FP"] + model_gnb["FN"])
model_gnb["recall"] = model_gnb["TP"] / float(model_gnb["TP"] + model_gnb["FN"])
model_gnb["specificity"] = model_gnb["TN"] / float(model_gnb["TN"] + model_gnb["FP"])
model_gnb["precision"] = model_gnb["TP"] / float(model_gnb["TP"] + model_gnb["FP"])

print("\n Model performance calculate from confusion matrix:\n" )
print("accuracy: ", model_gnb["accuracy"] )
print("recall: ",model_gnb["recall"] )
print("specificity: ",model_gnb["specificity"] )
print("precision: ",model_gnb["precision"] )

print ( """\n model_gnb["class_report"]""" )
print ( model_gnb["class_report"] )


"""
======================================
Compare result of 4 models
======================================
"""

print( "\n ===Logistic Regression (Baseline)===" )
print( "accuracy = ", model_logr_baseline["accuracy"] )
print( "recall = ", model_logr_baseline["recall"] )
print( "specificity = ", model_logr_baseline["specificity"] )
print( "precision = ", model_logr_baseline["precision"] )
print( "auc = ", model_logr_baseline["auc"] )

print( "\n ===Logistic Regression===" )
print( "accuracy = ", model_logr["accuracy"] )
print( "recall = ", model_logr["recall"] )
print( "specificity = ", model_logr["specificity"] )
print( "precision = ", model_logr["precision"] )
print( "auc = ", model_logr["auc"] )

print( "\n ===Random Forest===" )
print( "accuracy = ", model_rf["accuracy"] )
print( "recall = ", model_rf["recall"] )
print( "specificity = ", model_rf["specificity"] )
print( "precision = ", model_rf["precision"] )
print( "auc = ", model_rf["auc"] )

print( "\n ===Gaussian Naive Bayes===" )
print( "accuracy = ", model_gnb["accuracy"] )
print( "recall = ", model_gnb["recall"] )
print( "specificity = ", model_gnb["specificity"] )
print( "precision = ", model_gnb["precision"] )
print( "auc = ", model_gnb["auc"] )

"""
======================================
Plot ROC to compare 3 models
======================================
"""

fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(model_logr_baseline["roc"][0], model_logr_baseline["roc"][1], color="black", linestyle="--", dashes=(5, 1), label="Logistic Regression (Baseline), auc="+ str(model_logr_baseline["auc"]))
ax.plot(model_logr["roc"][0], model_logr["roc"][1], color="blue", label="Logistic Regression, auc="+ str(model_logr["auc"]))

ax.plot(model_rf["roc"][0], model_rf["roc"][1], color="green", label="Random Forest, auc="+ str(model_rf["auc"]))
ax.plot(model_gnb["roc"][0], model_gnb["roc"][1], color="red", label="Gaussian Naive Bayes, auc="+ str(model_gnb["auc"]))

ax.plot(ax.get_ylim(), ax.get_xlim(), color="gray", linewidth=0.5)

ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_xlim(left=0, right=1)
ax.set_ylim(bottom=0, top=1)
plt.legend(loc=4)
plt.show()