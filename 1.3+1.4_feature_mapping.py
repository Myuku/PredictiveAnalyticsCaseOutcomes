import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
def main():
    train_cases = read_data('./data/cases_2021_train_processed_2.csv')
    test_cases = read_data('./data/cases_2021_test_processed_unlabelled_2.csv')
    
    print(train_cases['chronic_disease_binary'].unique())
    # 0: deceased, 1: hospitalized, 2: non_hospitalized.
    # unique() == ['hospitalized' 'nonhospitalized' 'deceased']
    train_cases['outcome_group'].replace(to_replace = train_cases['outcome_group'].unique(),
                                         value=[1, 2, 0], inplace=True)
    # 0: Female, 1: Male
    # unique() == ['female' 'male']
    train_cases['sex'].replace(to_replace = train_cases['sex'].unique(), value=[0, 1], inplace=True)
    # 0: False, 1: True
    # unique() == [False  True]
    train_cases['chronic_disease_binary'].replace(to_replace = train_cases['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)

    # TODO: Add for country and province
    # train_cases['country']
    
    # For test_cases as well
    test_cases['sex'].replace(to_replace = test_cases['sex'].unique(), value=[0, 1], inplace=True)
    test_cases['chronic_disease_binary'].replace(to_replace = test_cases['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)
    
    print('train_cases: \n', train_cases)
    print('test_cases: \n', test_cases)
    
    # Write out as clean data file
    # train_cases.to_csv('./clean_cases/train_cases.csv', index=False)
    # test_cases.to_csv('./clean_cases/test_cases.csv', index=False)
    return
    
if __name__ == '__main__':
    main()