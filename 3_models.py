import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
# Reference: https://medium.com/dvt-engineering/hyper-parameter-tuning-for-scikit-learn-ml-models-860747bc3d72
def halving_grid_search_tuning(savefilename, model, x_train, y_train, params: dict):
    # svc = LinearSVC()
    # svc.fit(x_train, y_train)
    
    tuning_results = HalvingGridSearchCV(model, params, cv=10, factor=3, min_resources='exhaust').fit(x_train, y_train)
    
    results = pd.DataFrame(tuning_results.cv_results_)
    results.loc[:, 'mean_test_score'] *= 100
    results.to_csv('./model_results/'+savefilename+'.csv')
    
    res = results.loc[:, ('iter', 'rank_test_score', 'mean_test_score', 'params')]
    # res.sort_values(by=['iter', 'rank_test_score'], ascending=[False, True], inplace=True)
    return res
    
def linear_svc(x_train, y_train):
    svc = LinearSVC()
    params = {"a": [], "b": []}
    result = halving_grid_search_tuning('linear_svc', svc, x_train=x_train, y_train=y_train, params=params)
    return result
    
def main():
    train_cases = read_data('./clean_data/merged_data.csv')
    test_cases = read_data('./clean_data/cases_2021_test.csv')
    
    X = train_cases[~train_cases['outcome']].values
    y = train_cases['outcome'].values
    
    # Do a 80/20 train test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    
    
    # Test Model
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=0)
    mlp.fit(x_train, y_train)
    predictions = mlp.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    print('Accuracy: %.2f' % acc)
    
    
    
if __name__ == '__main__':
    main()