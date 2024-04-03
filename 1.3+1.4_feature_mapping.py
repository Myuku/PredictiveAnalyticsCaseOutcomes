import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
def main():
    train_cases = read_data('./all_data/partB/data/cases_2021_train_processed_2.csv')
    test_cases = read_data('./all_data/partB/data/cases_2021_test_processed_unlabelled_2.csv')
    
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
    
    # TODO: Need further discussion regarding on the needs of date_comfirmation
    # drop the date column for now
    train_cases.drop('date_confirmation', axis=1, inplace=True)
    test_cases.drop('date_confirmation', axis=1, inplace=True)

    # TODO: Add for country and province
    # Factorize 'country' and 'province' to encode them as integers
    le = LabelEncoder()
    train_cases['country'] = le.fit_transform(train_cases['country'])
    train_cases['province'] = le.fit_transform(train_cases['province'])

    # For test_cases as well
    test_cases['sex'].replace(to_replace = test_cases['sex'].unique(), value=[0, 1], inplace=True)
    test_cases['chronic_disease_binary'].replace(to_replace = test_cases['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)
    test_cases['country'] = le.fit_transform(test_cases['country'])
    test_cases['province'] = le.fit_transform(test_cases['province'])

    print('train_cases: \n', train_cases)
    print('test_cases: \n', test_cases)
    
    # Write out as clean data file
    train_cases.to_csv('./all_data/partB/clean_data/train_cases.csv', index=False)
    test_cases.to_csv('./all_data/partB/clean_data/test_cases.csv', index=False)
    return
    
if __name__ == '__main__':
    main()