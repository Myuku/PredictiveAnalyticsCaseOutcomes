import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data

def clean_cases(cases : pd.DataFrame):
    ''' Various values found from train cases
    'Hospitalized' 'Recovered' 'Deceased' 'Discharged' 'Alive' 'Discharge'
    'Under treatment' 'Stable' 'Died' 'Receiving treatment' 'Death'
    'Stable condition' 'Dead' 'Discharged from hospital' 'Critical condition'
    'Released from quarantine' 'Recovering at home 03.03.2020'
    '''
    hospitalized = ['Hospitalized', 'Discharged', 'Discharge', 'Under treatment', 
                    'Receiving treatment', 'Discharged from hospital', 'Critical condition']
    non_hospitalized = ['Alive', 'Stable', 'Stable condition', 'Released from quarantine', 
                        'Recovering at home 03.03.2020', 'Recovered']
    deceased = ['Deceased', 'Died', 'Death', 'Dead']
    
    cases['outcome'] = cases['outcome'].str.capitalize()
    cases.loc[cases['outcome'].isin(hospitalized), 'outcome'] = 'Hospitalized'
    cases.loc[cases['outcome'].isin(non_hospitalized), 'outcome'] = 'Non-Hospitalized'
    cases.loc[cases['outcome'].isin(deceased), 'outcome'] = 'Deceased'
    # Handle cases where there is more cases than what is known. Can't use 'Other' due to project req.
    cases.loc[~cases['outcome'].isin(['Hospitalized', 'Non-Hospitalized', 'Deceased']), 'outcome'] = 'Non-Hospitalized'
    return

    
def main():
    location_data = read_data('./data/location_2021.csv')
    train_cases = read_data('./data/cases_2021_train.csv')
    test_cases = read_data('./data/cases_2021_test.csv')
    
    # Add columns as per needed
    location_data = location_data[['Country_Region', 'Confirmed', 'Deaths', 'Province_State']]
    # age,sex,province,country,latitude,longitude,date_confirmation,additional_information,source,chronic_disease_binary
    train_cases = train_cases[['age', 'sex', 'country', 'province', 'chronic_disease_binary', 'outcome', 'outcome_group']]
    test_cases = test_cases[['age', 'sex', 'country', 'province', 'chronic_disease_binary', 'outcome_group']]
    print(train_cases)
    
    
    # standardize naming convention
    clean_cases(train_cases)
    location_data = location_data.rename(columns={'Country_Region': 'country', 'Province_State': 'province'}) 
    location_data.loc[location_data['country'] == 'US', 'country'] = 'United States'
    
    # Fix missing data
    location_data['province'] = location_data['province'].fillna(location_data['country'])
    
    # Write out as clean data file
    location_data.to_csv('./clean_data/location_2021.csv', index=False)
    train_cases.to_csv('./clean_data/cases_2021_train.csv', index=False)
    test_cases.to_csv('./clean_data/cases_2021_test.csv', index=False)
    return
    
if __name__ == '__main__':
    main()