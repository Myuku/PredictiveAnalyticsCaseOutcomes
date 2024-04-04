import pandas as pd
import numpy as np
import csv
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

RANDOM_STATE = 42
K_FOLD = 10         # can change, value has to be justified in report

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
# Accuracy on y_test: 0.33 (note: not using final train dataset)
def model_nn(x_train, x_test, y_train, y_test, params: dict):
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,), random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    
    acc = accuracy_score(y_test, y_preds)
    rep = classification_report(y_test, y_preds)
    
    print('Neural Network Accuracy: %.2f' % acc)
    print('Report: \n', rep)
    
    
    # With hyperparam tuning method - Not tested yet
    # tuning_results = HalvingGridSearchCV(clf, params, cv=K_FOLD, factor=3, min_resources='exhaust').fit(x_train, y_train)
    # results = pd.DataFrame(tuning_results.cv_results_)
    # results.loc[:, 'mean_test_score'] *= 100
    # results.to_csv('./model_results/%s.csv' % 'nn')
    return
    
# Accuracy on y_test: 0.90 (note: not using final train dataset)
def model_rf(x_train, x_test, y_train, y_test, params: dict):
    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    
    train_preds = clf.predict(x_train)
    train_acc = accuracy_score(y_train, train_preds)
    y_preds = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_preds)
    
    rep = classification_report(y_test, y_preds, zero_division = 1)
    
    # plot_roc_model(clf, x_test, y_test)
    _, _, train_roc_auc, _ = calc_roc_auc(clf, x_train, y_train)
    _, _, test_roc_auc, _ = calc_roc_auc(clf, x_test, y_test)
    avg_train_auc = np.mean(list(train_roc_auc.values()))
    avg_test_auc = np.mean(list(test_roc_auc.values()))

    return train_acc, test_acc, rep, avg_train_auc, avg_test_auc

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
    criterions = ['gini', 'entropy'] # No effect
    tune_hparam_auc_acc(model_rf, x_train, x_test, y_train, y_test,
                     criterions, {'n_estimators': 40, 'random_state': RANDOM_STATE}, 'criterion')
    return
    
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

    # rf_params = {'n_estimators': [1,2,3,4,5,6,7,8,9,10], 'criterion': ['gini', 'entropy'], 
    #              'min_samples_split': [2,3,4,5], 'min_samples_leaf': [1,2,3,4,5], 
    #              'max_features': ['sqrt', 'log2', len(train_cases.drop('outcome_group', axis=1).columns)-1]}
    
    # Smaller max_features reduces overfitting; sqrt is best for classification generally
    # https://datascience.stackexchange.com/questions/66825/how-many-features-does-random-forest-need-for-the-trees
    rf_params = {'n_estimators': 40, 'criterion': 'gini', 
                 'min_samples_split': 2, 'min_samples_leaf': 1,  
                 'max_features': 'sqrt', 'max_depth': 20,
                 'random_state': RANDOM_STATE}
    
    print("\nWithout Scalers(): ")
    train_acc, test_acc, rep, _, _ = model_rf(x_train, x_test, y_train, y_test, rf_params)
    print('Random Forest Train Accuracy: %.2f' % train_acc)
    print('Random Forest Test Accuracy: %.2f' % test_acc)
    print('Report: \n', rep) 
    
    
    # TODO: Scalers don't work well because we need to identify which ones need to be scaled and not.
    #       It is currently having a negative effect.
    # print("\nWith MinMaxScaler(): ")
    # model_rf(x_train_minmax, x_test_minmax, y_train, y_test, rf_params)
    # print("\nWith StandardScaler(): ")  
    # model_rf(x_train_standard, x_test_standard, y_train, y_test, rf_params)
    
    
    # TODO: Hyperparameter tuning with GridSearchCV, RandomSearchCV, BayesianOptimization
    
    # "For this project, the goal is to predict the outcome (hospitalized, non-hospitalized, deceased)
    # of a case correctly with a small number of false negatives and false positives"
    
    # Determined ROC curves to obtain good False Positive / True Positive rates 
    # He means F1 precision and recall probably, but ROC can work as well
    # Graphing purposes!!! - ROC curves can be seen if you uncomment it in the used model
    # Comment out to reduce processing time
    plot_auc_rf_tuning(model_rf, x_train, x_test, y_train, y_test)
    

    # TODO: Best model with best parameters
    # best_model = ...
    
    
    # Task 9: Prediction on test set
    # model_name = ""
    # predictions = ...
    # create_submission_file(y_preds=predictions, "submission_%s.csv" % model_name)
    return
    
    
if __name__ == '__main__':
    main()