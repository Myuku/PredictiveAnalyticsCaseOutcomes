import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
    
def main():
    train_cases = read_data('./all_data/partB/clean_data/balanced_test_cases.csv')
    test_cases = read_data('./all_data/partB/clean_data/cases_2021_test_processed_2.csv')
    
    X = train_cases[~train_cases['outcome']].values
    y = train_cases['outcome'].values
    
    # Do a 80/20 train test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    
    
    
if __name__ == '__main__':
    main()