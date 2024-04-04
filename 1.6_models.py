import pandas as pd
import csv
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
    print('Accuracy: %.2f' % acc)
    print('Report: \n', rep)
    
    
    # With hyperparam tuning method - Not tested yet
    # tuning_results = HalvingGridSearchCV(clf, params, cv=K_FOLD, factor=3, min_resources='exhaust').fit(x_train, y_train)
    # results = pd.DataFrame(tuning_results.cv_results_)
    # results.loc[:, 'mean_test_score'] *= 100
    # results.to_csv('./model_results/%s.csv' % 'nn')
    return
    
# Accuracy on y_test: 0.90 (note: not using final train dataset)
def model_rf(x_train, x_test, y_train, y_test, params: dict):
    clf = RandomForestClassifier(n_estimators=5, criterion="gini", random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    acc = accuracy_score(y_test, y_preds)
    rep = classification_report(y_test, y_preds)
    print('Accuracy: %.2f' % acc)
    print('Report: \n', rep)
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
    
    ## Task 6: Run models
    # model_nn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, params={})
    
    rf_params = {'n_estimators': [1,2,3,4,5,6,7,8,9,10], 'criterion': ['gini', 'entropy'], 'min_sample_split': [2,3,4,5], 'min_samples_leaf': [1,2,3,4,5], 'max_features': ['sqrt', 'log2', len(train_cases.drop('outcome_group', axis=1).columns)-1]}
    model_rf(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, params=rf_params)
    
    
    # TODO: Best model with best parameters
    # best_model = ...
    
    
    # Task 9: Prediction on test set
    # model_name = ""
    # predictions = ...
    # create_submission_file(y_preds=predictions, "submission_%s.csv" % model_name)
    return
    
    
if __name__ == '__main__':
    main()