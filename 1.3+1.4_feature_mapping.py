import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
def main():
    train_cases = read_data('./data/cases_2021_train_processed_unlabelled_2.csv')
    test_cases = read_data('./data/cases_2021_test_processed_unlabelled_2.csv')
    
    # TODO: Need to redo with the new dataset
    
    # 0: deceased, 1: hospitalized, 2: non_hospitalized.
    # unique() == ['Hospitalized' 'Non-Hospitalized' 'Deceased']
    train_cases['outcome'].replace(to_replace = train_cases['outcome'].unique(), value=[1, 2, 0], inplace=True)
    train_cases['outcome_group'] = train_cases['outcome']
    # -1: nan, 0: Female, 1: Male
    # unique() == [nan 'female' 'male']
    train_cases['sex'].replace(to_replace = train_cases['sex'].unique(), value=[-1, 0, 1], inplace=True)
    # 0: False, 1: True
    # unique() == [False  True]
    train_cases['chronic_disease_binary'].replace(to_replace = train_cases['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)
    train_cases['country']
    
    # For test_cases as well
    test_cases['sex'].replace(to_replace = test_cases['sex'].unique(), value=[-1, 0, 1], inplace=True)
    test_cases['chronic_disease_binary'].replace(to_replace = test_cases['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)
    
    print('train_cases: \n', train_cases)
    print('test_cases: \n', test_cases)
    
    # ... add on
    
    # Write out as clean data file
    train_cases.to_csv('./clean_cases/train_cases.csv', index=False)
    test_cases.to_csv('./clean_cases/test_cases.csv', index=False)
    return
    
if __name__ == '__main__':
    main()