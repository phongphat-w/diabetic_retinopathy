# -*- coding: utf-8 -*-
"""
@author: Phongphat Wiwatthanasetthakarn
"""

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB


from scipy.stats import uniform


class Experiment():
    
    def __init__(self, X=None, y=None, base_estimator=None):
        
        self.model_return = {}
        
        self.X_train = X
        self.y_train = y  
        
        self.base_estimator = base_estimator
    
    
    def logistic_regression(self):        
        print ( "\n logistic_regression() is activated...\n" )
    
        model_logr = {}
        
        model_logr["hparams"] = {"C": uniform(loc=0, scale=4)}
        model_logr["model"] = LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced")
        model_logr["rcv"] = RandomizedSearchCV(model_logr["model"], model_logr["hparams"], random_state=0)
        model_logr["rsearch"] = model_logr["rcv"].fit(self.X_train, self.y_train)
        
        """Finding the best hyperparameters """        
        model_logr["rtuned_hparams"] = model_logr["rsearch"].best_params_
        
        
        """Grid search """        
        C = model_logr["rtuned_hparams"]["C"]
        model_logr["gparams"] = {"C": [C - (C*i/10) for i in range(5)] + [C + (C*i/10) for i in range(1, 5)]}
        model_logr["gcv"] = GridSearchCV(model_logr["model"], model_logr["gparams"])
        model_logr["gcv"].fit(self.X_train, self.y_train)
        model_logr["tuned_model"] = model_logr["gcv"].best_estimator_
        
        return model_logr
    
    
    def random_forest(self):
        print ( "\n random_forest() is activated...\n" )
        
        model_rf = {}

        model_rf["hparams"] = {"max_depth": [2, 4, 6], "min_samples_split": [5, 10]}
        model_rf["model"] = RandomForestClassifier(random_state=0, class_weight="balanced")
        model_rf["rcv"] = RandomizedSearchCV(model_rf["model"], model_rf["hparams"], random_state=0)
        model_rf["rsearch"] = model_rf["rcv"].fit(self.X_train, self.y_train)
        
        """Finding the best hyperparameters """
        model_rf["rtuned_hparams"] = model_rf["rsearch"].best_params_
        
        """Grid search """
        max_depth = model_rf["rtuned_hparams"]["max_depth"]
        min_samples_split = model_rf["rtuned_hparams"]["min_samples_split"]
        model_rf["gparams"] = {"max_depth": [max_depth - 1, max_depth, max_depth + 1], 
                         "min_samples_split": [min_samples_split - 1, min_samples_split, min_samples_split + 1]}
        model_rf["gcv"] = GridSearchCV(model_rf["model"], model_rf["gparams"], cv=10)
        model_rf["gcv"].fit(self.X_train, self.y_train)
        model_rf["tuned_model"] = model_rf["gcv"].best_estimator_

        return model_rf


    def gnb(self):
        print ( "\n gnb() is activated...\n" )
        
        model_gnb = {}
        
        model_gnb["model"] = GaussianNB()
        model_gnb["tuned_model"] = model_gnb["model"].fit(self.X_train, self.y_train)
                
        return model_gnb
    
   
