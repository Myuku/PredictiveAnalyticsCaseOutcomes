import pandas as pd
import numpy as np
import csv
import json
import ast
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split, GridSearchCV, RandomizedSearchCV, validation_curve, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, make_scorer
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy.stats import uniform, randint
from sklearn.svm import SVC
import xgboost as xgb
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout


RANDOM_STATE = 42
K_FOLD = 5         # DEFAULT IS 5
TARGET_NAMES =['deceased', 'hospitalized', 'non_hospitalized']

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data

''' MODELS '''
# Accuracy on y_test: 0.33 (note: not using final train dataset)        --- might skip nn model as a whole due to poor performance
# def model_nn(x_train, x_test, y_train, y_test, params: dict):
#     clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,), random_state=RANDOM_STATE)
#     clf.fit(x_train, y_train)
#     y_preds = clf.predict(x_test)
    
#     acc = accuracy_score(y_test, y_preds)
#     rep = classification_report(y_test, y_preds)
    
#     print('Neural Network Accuracy: %.2f' % acc)
#     print('Report: \n', rep)
    
    
#     # With hyperparam tuning method - Not tested yet
#     # tuning_results = HalvingGridSearchCV(clf, params, cv=K_FOLD, factor=3, min_resources='exhaust').fit(x_train, y_train)
#     # results = pd.DataFrame(tuning_results.cv_results_)
#     # results.to_csv('./model_results/%s.csv' % 'nn')
#     return
    
# Accuracy on y_test: 0.90 
def model_rf(x_train, x_test, y_train, y_test, params: dict, hyper_tuning: bool = False):
    
    if not hyper_tuning:
        clf = RandomForestClassifier(**params)
        clf.fit(x_train, y_train)
    
        train_preds = clf.predict(x_train)
        train_acc = accuracy_score(y_train, train_preds)
        y_preds = clf.predict(x_test)
        test_acc = accuracy_score(y_test, y_preds)
        
        rep = classification_report(y_test, y_preds, zero_division = 1, output_dict=True, target_names=TARGET_NAMES)
        results = {
            'hyperparameters': str(params),
            'mean_macro_f1': f"{rep['macro avg']['f1-score']: .2f}",
            'mean_deceased_f1': f"{rep['deceased']['f1-score']: .2f}",
            'mean_overall_accuracy': f"{test_acc: .2f}"
        }
    
        # plot_roc_model(clf, x_test, y_test)
        _, _, train_roc_auc, _ = calc_roc_auc(clf, x_train, y_train)
        _, _, test_roc_auc, _ = calc_roc_auc(clf, x_test, y_test)
        avg_train_auc = np.mean(list(train_roc_auc.values()))
        avg_test_auc = np.mean(list(test_roc_auc.values()))
        return train_acc, test_acc, rep, results, avg_train_auc, avg_test_auc
    else:
        clf = RandomForestClassifier()
        tuning = RandomizedSearchCV(clf, param_distributions=params, random_state=RANDOM_STATE, n_iter=50, cv=5, verbose=1, return_train_score=True)
        tuning.fit(x_train, y_train)
        results = pd.DataFrame(tuning.cv_results_)
        results.to_csv('./all_data/partB/model_results/%s.csv' % 'rf')
        
        train_preds = tuning.predict(x_train)
        train_acc = accuracy_score(y_train, train_preds)
        y_preds = tuning.predict(x_test)
        test_acc = accuracy_score(y_test, y_preds)
        rep = classification_report(y_test, y_preds, zero_division = 1, output_dict=True, target_names=TARGET_NAMES)
        # best_score: average cross-validated score
        return train_acc, test_acc, rep, tuning.best_score_, tuning.best_params_
    
    
# Linear SVC is better than SVC for large data sets
# TODO: to be tested
def model_linearsvc(x_train, x_test, y_train, y_test, params: dict, hyper_tuning: bool = False):
    if not hyper_tuning:
        clf = LinearSVC(**params)
        clf.fit(x_train, y_train)
    
        train_preds = clf.predict(x_train)
        train_acc = accuracy_score(y_train, train_preds)
        y_preds = clf.predict(x_test)
        test_acc = accuracy_score(y_test, y_preds)
        
        rep = classification_report(y_test, y_preds, zero_division = 1)
        
        return train_acc, test_acc
    else:
        clf = LinearSVC()

# TODO: to be tested
# XGBoost Test Accuracy: 0.88 (Default params)
# XGBoost Test Accuracy: 0.9 (with hyper tuning)
def model_xgboost(x_train, x_test, y_train, y_test, params: dict, hyper_tuning: bool = False):
    if not hyper_tuning:
        clf = xgb.XGBClassifier(objective="multi:softprob", random_state=RANDOM_STATE, **params)      # for multi-class classification
        clf.fit(x_train, y_train)
    
        train_preds = clf.predict(x_train)
        train_acc = accuracy_score(y_train, train_preds)
        y_preds = clf.predict(x_test)
        test_acc = accuracy_score(y_test, y_preds)
        
        rep = classification_report(y_test, y_preds, zero_division = 1, output_dict=True, target_names=TARGET_NAMES)
        results = {
            'hyperparameters': str(params),
            'mean_macro_f1': f"{rep['macro avg']['f1-score']: .2f}",
            'mean_deceased_f1': f"{rep['deceased']['f1-score']: .2f}",
            'mean_overall_accuracy': f"{test_acc: .2f}"
        }
        
        return train_acc, test_acc, rep, results, y_preds
    else:
        clf = xgb.XGBClassifier(objective="multi:softprob", random_state=RANDOM_STATE)
        tuning = RandomizedSearchCV(clf, param_distributions=params, random_state=RANDOM_STATE, n_iter=50, cv=5, verbose=1, n_jobs=1, return_train_score=True, scoring='accuracy')
        tuning.fit(x_train, y_train)
        results = pd.DataFrame(tuning.cv_results_)
        results.to_csv('./all_data/partB/model_results/%s.csv' % 'xgb')
        
        train_preds = tuning.predict(x_train)
        train_acc = accuracy_score(y_train, train_preds)
        y_preds = tuning.predict(x_test)
        test_acc = accuracy_score(y_test, y_preds)
        rep = classification_report(y_test, y_preds, zero_division = 1, output_dict=True, target_names=TARGET_NAMES)
        return train_acc, test_acc, rep, tuning.best_score_, tuning.best_params_

# model svm
def model_svm(x_train, x_test, y_train, y_test, svm_params, use_scale='none'):
    # Apply scaling if specified
    if use_scale == 'minmax':
        scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    elif use_scale == 'standard':
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    svm_model = SVC(**svm_params)

    svm_model.fit(x_train, y_train)

    y_train_pred = svm_model.predict(x_train)
    y_test_pred = svm_model.predict(x_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    rep = classification_report(y_test, y_test_pred, zero_division = 1, output_dict=True, target_names=TARGET_NAMES)

    return train_acc, test_acc, rep

def model_knn(x_train, x_test, y_train, y_test, params: dict, hyper_tuning: bool = False):
    
    if not hyper_tuning:
        clf = KNeighborsClassifier(**params)
        clf.fit(x_train, y_train)
    
        train_preds = clf.predict(x_train)
        train_acc = accuracy_score(y_train, train_preds)
        y_preds = clf.predict(x_test)
        test_acc = accuracy_score(y_test, y_preds)
        
        rep = classification_report(y_test, y_preds, zero_division = 1, output_dict=True, target_names=TARGET_NAMES)
        results = {
            'hyperparameters': str(params),
            'mean_macro_f1': f"{rep['macro avg']['f1-score']: .2f}",
            'mean_deceased_f1': f"{rep['deceased']['f1-score']: .2f}",
            'mean_overall_accuracy': f"{test_acc: .2f}"
        }
        return train_acc, test_acc, rep, results
    
    else:
        clf = KNeighborsClassifier()
        # tuning = RandomizedSearchCV(clf, param_distributions=params, random_state=RANDOM_STATE, 
                                    # n_iter=10, cv=5, verbose=2, n_jobs=1, return_train_score=True)
        tuning = GridSearchCV(clf, param_grid=params, refit=True, verbose=3, n_jobs=-1)   
        
        tuning.fit(x_train, y_train)
        results = pd.DataFrame(tuning.cv_results_)
        results.to_csv('./all_data/partB/model_results/%s.csv' % 'knn')
        
        train_preds = tuning.predict(x_train)
        train_acc = accuracy_score(y_train, train_preds)
        y_preds = tuning.predict(x_test)
        test_acc = accuracy_score(y_test, y_preds)
        rep = classification_report(y_test, y_preds, zero_division = 1, output_dict=True, target_names=TARGET_NAMES)
        # best_score: average cross-validated score
        return train_acc, test_acc, rep, tuning.best_score_, tuning.best_params_


''' PLOTS & ANALYSIS '''
# https://stackoverflow.com/questions/51378105/plot-multi-class-roc-curve-for-decisiontreeclassifier
# Credits to drew_psy 
# Calculating False Positive / True Positive rates
def calc_roc_auc(clf, x, y):
    y_proba = clf.predict_proba(x)
    y_bin = label_binarize(y, classes = [0, 1, 2])
    n_classes = y_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc, n_classes
        
def plot_roc_model(clf, x, y):
    fpr, tpr, roc_auc, n_classes = calc_roc_auc(clf, x, y)
        
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()
    return 

def tune_hparam_auc_acc(model, x_train, x_test, y_train, y_test, values, params, param: str):
    train_results = []
    test_results = []
    train_accuracies = []
    test_accuracies = []
    
    for value in values:
        params[param] = value
        train_acc, test_acc, _, avg_train_auc, avg_test_auc = model(x_train, x_test, y_train, y_test, params)
        train_results.append(avg_train_auc)
        test_results.append(avg_test_auc)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
   
    values = ['None' if v is None else v for v in values]
    plt.figure()
    line1, = plt.plot(values, train_results, 'b', label="Train AUC")
    line2, = plt.plot(values, test_results, 'r', label="Test AUC")
    line3, = plt.plot(values, train_accuracies, 'g', label="Train Acc")
    line4, = plt.plot(values, test_accuracies, 'y', label="Test Acc")
    
    plt.legend(handler_map = {line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC/Acc score')
    plt.xlabel(param)
    plt.title('Different Values of ' + param + ' on Accuracy and AUC')
    plt.savefig('./all_data/partB/plots/auc_acc_' + param + '.png')
    
def plot_auc_rf_tuning(rfmodel, x_train, x_test, y_train, y_test):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 128] # >40 is good
    tune_hparam_auc_acc(model_rf, x_train, x_test, y_train, y_test,
                     n_estimators, {'random_state': RANDOM_STATE}, 'n_estimators')
    max_depths = range(1, 32) # >20 is good
    tune_hparam_auc_acc(model_rf, x_train, x_test, y_train, y_test,
                     max_depths, {'n_estimators': 40, 'random_state': RANDOM_STATE}, 'max_depth')
    max_features = ['sqrt', 'log2', len(x_train[0]), None] # sqrt is best
    tune_hparam_auc_acc(model_rf, x_train, x_test, y_train, y_test,
                     max_features, {'n_estimators': 40, 'random_state': RANDOM_STATE}, 'max_features')
    # 1% to 50% of total data
    min_samples_splits = np.linspace(0.01, 0.5, 50, endpoint=True) # Lower is better (Only raise if not enough time/resources)
    tune_hparam_auc_acc(model_rf, x_train, x_test, y_train, y_test,
                     min_samples_splits, {'n_estimators': 40, 'random_state': RANDOM_STATE}, 'min_samples_split')
    # 1% to 50% of total data
    min_samples_leafs = np.linspace(0.01, 0.5, 50, endpoint=True) # Lower is better (Only raise if not enough time/resources)
    tune_hparam_auc_acc(model_rf, x_train, x_test, y_train, y_test,
                     min_samples_leafs, {'n_estimators': 40, 'random_state': RANDOM_STATE}, 'min_samples_leaf')
    criterions = ['gini', 'entropy', 'log_loss'] # Gini is best
    tune_hparam_auc_acc(model_rf, x_train, x_test, y_train, y_test,
                     criterions, {'n_estimators': 40, 'random_state': RANDOM_STATE}, 'criterion')
    return

def plot_knn_k_tuning(X, y):
    
    param_range = range(1, 11)
    # Calculate accuracy on training and test set using the
    # gamma parameter with 5-fold cross validation
    train_score, test_score = validation_curve(KNeighborsClassifier(), X, y, param_name = "n_neighbors",
                                               param_range = param_range, cv = 5, scoring = "accuracy")
        
    plt.figure()
    line1, = plt.plot(param_range, np.mean(train_score, axis = 1), 'r', label = "Train Accuracy")
    line2, = plt.plot(param_range, np.mean(test_score, axis = 1), 'b', label = "Cross Validation (Test) Accuracy")
    plt.legend(handler_map = {line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    plt.xlabel('K Value')
    plt.title('Validation Curve of KNN Classifier')
    plt.savefig('./all_data/partB/plots/knn_validation_curve.png')

''' Get f1 scores from hyperparams '''
def get_scores(path: str, model, x_train, x_test, y_train, y_test):
    model_tuning_data = read_data(path)
    model_tuning_data = model_tuning_data[['params']]
    params = model_tuning_data['params'].tolist()
    
    df = pd.DataFrame(columns=['hyperparameters', 'mean_macro_f1', 'mean_deceased_f1', 'mean_overall_accuracy'])
    for param in params:
        p = ast.literal_eval(param)         # convert str dictionary to dictionary
        test_acc = None
        report = None
        if model == 'xgb':
            _, test_acc, report, results, _ = model_xgboost(x_train, x_test, y_train, y_test, p)
        elif model == 'knn':
            _, test_acc, report, results = model_knn(x_train, x_test, y_train, y_test, p)
        elif model == 'randomforest':
            _, test_acc, report, results, _, _ = model_rf(x_train, x_test, y_train, y_test, p)
            
        # Get the scores
        # new_row = {
        #     'hyperparameters': param,
        #     'mean_macro_f1': f"{report['macro avg']['f1-score']: .2f}",
        #     'mean_deceased_f1': f"{report['deceased']['f1-score']: .2f}",
        #     'mean_overall_accuracy': f"{test_acc: .2f}"
        # }
        df.loc[len(df)] = results       # add new row
    
    df.to_csv('./results/%s_tuning.txt' % model, index=False)
    
'''For labelling test data without predictions of the outcome_group'''
def create_submission_file(y_preds, file_name):
    with open(file_name, "w") as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id", "Prediction"])
        for i, pred in enumerate(y_preds):
            wr.writerow([str(i), str(pred)])
    return

def main():
    train_cases = read_data('./all_data/partB/balanced_data/train_cases.csv')
    test_cases = read_data('./all_data/partB/data/cases_2021_test_processed_unlabelled_2.csv')
    
    X = train_cases.drop('outcome_group', axis=1).values
    y = train_cases['outcome_group'].values
    
    # Do a 80/20 train test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # TODO: Futher preprocessing if required e.g. MinMaxScalar, etc...
    min_max_scaler = MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    x_test_minmax = min_max_scaler.fit_transform(x_test)
    
    standard_scaler = StandardScaler()
    x_train_standard = standard_scaler.fit_transform(x_train)
    x_test_standard = standard_scaler.fit_transform(x_test)
    
    ## Task 6: Run models
    # model_nn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, params={})

    '''Model 1: Random Forest'''
    # Smaller max_features reduces overfitting; sqrt is best for classification generally
    # https://datascience.stackexchange.com/questions/66825/how-many-features-does-random-forest-need-for-the-trees
    rf_params = {'n_estimators': 40, 'criterion': 'gini', 
                 'min_samples_split': 2, 'min_samples_leaf': 1,  
                 'max_features': 'sqrt', 'max_depth': 20,
                 'random_state': RANDOM_STATE}
                 
    # Cut down on number of params possibilities else it will take a long time
    # rf_params_tuning = {'n_estimators': [32, 64], 
    #                     'criterion' :['gini', 'entropy'],
    #                      'min_samples_split': [0.01], 
    #                      'min_samples_leaf': [0.01, 0.1],  
    #                      'max_features': ['sqrt', 'log2', len(x_train[0])-1],
    #                      'max_depth': list(range(20,32)),
    #                      'random_state': [RANDOM_STATE]
    #                      }
    rf_params_tuning = {'n_estimators': list(range(30, 41)), 
                        'criterion' :['gini', 'entropy'],
                         'min_samples_split': [2,3,4,5], 
                         'min_samples_leaf': [1,2,3,4,5],  
                         'max_features': ['sqrt', 'log2'],
                         'max_depth': list(range(20, 26)),
                         'random_state': [RANDOM_STATE]
                         }

    
    # print("\n1. Without Scalers(): ")
    # train_acc, test_acc, rep, results, _, _ = model_rf(x_train, x_test, y_train, y_test, rf_params)
    # print('Random Forest Train Accuracy: %.2f' % train_acc)
    # print('Random Forest Test Accuracy: %.2f' % test_acc)
    # print('Report: \n', rep) 
    # print('Results: \n')
    # print('Hyperparamters: %s' % results['hyperparameters'])
    # print('Mean macro f1-score: %s' % results['mean_macro_f1'])
    # print('Mean deceased f1-score: %s' % results['mean_deceased_f1'])
    # print('Mean overall accuracy: %s' % results['mean_overall_accuracy'])
    
    # print("\n2. Without Scalers(), with Hyperparam tuning(): ")
    # train_acc, test_acc, rep, best_score, best_params = model_rf(x_train, x_test, y_train, y_test, rf_params_tuning, hyper_tuning=True)
    # print("Random Forest best params: ", best_params)
    # print('Random Forest Train Accuracy: %.2f' % train_acc)
    # print('Random Forest Test Accuracy: %.2f' % test_acc)
    # print('Random Forest Report: \n', rep) 
    
    
    ''''Model 2: XGBoost '''
    xgb_params = {
                    "colsample_bytree": 0.89,              
                    "gamma": 0.07,                           
                    "learning_rate": 0.12,                
                    "max_depth": 9,                         
                    "n_estimators": 370,                  
                    "subsample": 0.78,
                 }
    # For a good general understanding on main params for XGBoost: https://medium.com/@rithpansanga/the-main-parameters-in-xgboost-and-their-effects-on-model-performance-4f9833cac7c
    xgb_params_tuning = {
                    "colsample_bytree": np.linspace(0.5, 1, 100, endpoint=True),              # controls fraction of features used for each tree. smaller -> smaller and less complex models (prevents overfitting) common=[0.5, 1]
                    "gamma": np.linspace(0, 0.1, 100, endpoint=True),                           # 
                    "learning_rate": np.linspace(0.05, 0.3, 100, endpoint=True),                # smaller -> slower but more accurate, default=0.3  
                    "max_depth": randint(2, 10),                         # smaller -> simplier model (underfitting), larger -> overfitting. default=6
                    "n_estimators": randint(100, 500),                  # number of trees. larger --> overfitting. default 100. common=[100, 1000]
                    "subsample": np.linspace(0.7, 1, 100, endpoint=True)
                }

    # print("\n1. Without Scalers(): ")
    # train_acc, test_acc, rep, results, y_preds = model_xgboost(x_train, x_test, y_train, y_test, xgb_params)
    # print('XGBoost Train Accuracy: %.2f' % train_acc)
    # print('XGBoost Test Accuracy: %.2f' % test_acc)
    # print('XGBoost Report: \n', rep) 
    # print('Results: \n')
    # print('Hyperparamters: %s' % results['hyperparameters'])
    # print('Mean macro f1-score: %s' % results['mean_macro_f1'])
    # print('Mean deceased f1-score: %s' % results['mean_deceased_f1'])
    # print('Mean overall accuracy: %s' % results['mean_overall_accuracy'])
    
    # print("\n2. Without Scalers(), with Hyperparam tuning(): ")
    # train_acc, test_acc, rep, best_score, best_params = model_xgboost(x_train, x_test, y_train, y_test, xgb_params_tuning, hyper_tuning=True)
    # print("XGBoost best params: ", best_params)
    # print("XGBoost best score: ", best_score)
    # print('XGBoost Train Accuracy: %.2f' % train_acc)
    # print('XGBoost Test Accuracy: %.2f' % test_acc)
    # print('XGBoost Report: \n', rep) 
    

    ''''Model 3: SVM '''
    # svm_params = {
    # 'C': 1.0,  
    # 'kernel': 'linear',  
    # 'gamma': 'scale',  
    # 'random_state': RANDOM_STATE
    # }
    
    # print("\nSVM Model Evaluation: ")
    # train_acc, test_acc, rep = model_svm(x_train, x_test, y_train, y_test, svm_params, use_scale='standard')
    # print('SVM Train Accuracy: %.2f' % train_acc)
    # print('SVM Test Accuracy: %.2f' % test_acc)
    # print('SVM Classification Report: \n', rep)

    ''''Model 4: K-Nearest Neighbours '''
    
    # KNN Drawbacks, 
    knn_params = {'n_neighbors': 4, 
                  'weights' : 'distance', # Distance is is a lot more strict
                  'algorithm': 'kd_tree', 
                  'leaf_size': 25,  
                  'p': 1, # p = 1: manhattan_distance (l1), p = 2: euclidean_distance (l2)
                  'n_jobs': -1 # use all your cpu cores
                  }

    knn_params_tuning = {'n_neighbors': range(3, 10), 
                         'weights' :['uniform', 'distance'], # Distance is is a lot more strict
                         'algorithm': ['kd_tree', 'ball_tree'], # brute force is worse in every aspect
                         'leaf_size': [25],  # Has no effect on the model, only on the speed of execution
                         'p': [1, 2],
                        'n_jobs': [-1]
                         }

    # print("\n1. Without Scalers(): ")   
    # train_acc, test_acc, rep, results = model_knn(x_train, x_test, y_train, y_test, knn_params)
    # print('K-Nearest Train Accuracy: %.2f' % train_acc)
    # print('K-Nearest Test Accuracy: %.2f' % test_acc)
    # print('Report: \n', rep) 
    # print('Results: \n')
    # print('Hyperparamters: %s' % results['hyperparameters'])
    # print('Mean macro f1-score: %s' % results['mean_macro_f1'])
    # print('Mean deceased f1-score: %s' % results['mean_deceased_f1'])
    # print('Mean overall accuracy: %s' % results['mean_overall_accuracy'])
    
    print("\n2. Without Scalers(), with Hyperparam tuning(): ")
    train_acc, test_acc, rep, best_score, best_params = model_knn(x_train, x_test, y_train, y_test, knn_params_tuning, hyper_tuning=True)
    print("K-Nearest best params: ", best_params)
    print("K-Nearest best score: ", best_score)
    print('K-Nearest Train Accuracy: %.2f' % train_acc)
    print('K-Nearest Test Accuracy: %.2f' % test_acc)
    print('K-Nearest Report: \n', rep) 
    
    # plot_knn_k_tuning(X, y)
    
    
    
    ''' Save scores from hyperparameter tunings '''
    # get_scores('./all_data/partB/model_results/xgb.csv', 'xgb', x_train, x_test, y_train, y_test)
    # get_scores('./all_data/partB/model_results/rf.csv', 'randomforest', x_train, x_test, y_train, y_test)
    get_scores('./all_data/partB/model_results/knn.csv', 'knn', x_train, x_test, y_train, y_test)
    
    

    
    # TODO: Scalers don't work well because we need to identify which ones need to be scaled and not.
    #       It is currently having a negative effect.
    # print("\nWith MinMaxScaler(): ")
    # model_rf(x_train_minmax, x_test_minmax, y_train, y_test, rf_params)
    # print("\nWith StandardScaler(): ")  
    # model_rf(x_train_standard, x_test_standard, y_train, y_test, rf_params)
    
    
    # TODO: Hyperparameter tuning with GridSearchCV, RandomSearchCV, BayesianOptimization
    
    '''
    "For this project, the goal is to predict the outcome (hospitalized, non-hospitalized, deceased)
    of a case correctly with a small number of false negatives and false positives"
    
    Determined ROC curves to obtain good False Positive / True Positive rates 
    He means F1 precision and recall probably, but ROC can work as well
    Graphing purposes!!! - ROC curves can be seen if you uncomment it in the used model
    Comment out to reduce processing time
    '''
    # plot_auc_rf_tuning(model_rf, x_train, x_test, y_train, y_test)
    

    # TODO: Best model with best parameters
    # best_model = ...
    
    
    # Task 9: Prediction on test set
    # model_name = "xgb"
    
    # xgb_params = {
    #                 "colsample_bytree": 0.89,              
    #                 "gamma": 0.07,                           
    #                 "learning_rate": 0.12,                
    #                 "max_depth": 9,                         
    #                 "n_estimators": 370,                  
    #                 "subsample": 0.78,
    #              }
    # _, _, _, predictions = model_xgboost(x_train, x_test, y_train, y_test, params=xgb_params)
    # create_submission_file(y_preds=predictions, file_name="submission_%s.csv" % model_name)
    return
    
    
if __name__ == '__main__':
    main()