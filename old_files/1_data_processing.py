import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
def main():
    location_data = read_data('./clean_data/location_2021.csv')
    train_cases = read_data('./clean_data/cases_2021_train.csv')
    test_cases = read_data('./clean_data/cases_2021_test.csv')
    
    # Merge data
    merged_data = pd.merge(train_cases, location_data, how='left', on=['country', 'province'])        # Error note: some province are NaN in location_2021.csv so this is not working currently
    
    # Add Expected Mortality Rate
    # TODO: observed deaths in cases_2021_train (based on location) / Location (country, province) Deaths count
    
    observed_deaths_data = merged_data[merged_data['outcome'] == 'Deceased']    \
                           .value_counts(['country', 'province', 'outcome']).reset_index()  \
                           .rename(columns={"count": "observed_deaths"}).drop('outcome', axis=1)
    merged_data = pd.merge(merged_data, observed_deaths_data, how='left', on=['country', 'province']) 
    merged_data['Expected_Mortality_Rate'] = merged_data['observed_deaths'] / merged_data['Deaths']
    
    
    # FIXME: country and province needs to be converted too, but not sure how to handle it.
    # Map categorical features to numeric
    # Use '-1' for NaN values 
    
    # 0: deceased, 1: hospitalized, 2: non_hospitalized.
    # unique() == ['Hospitalized' 'Non-Hospitalized' 'Deceased']
    merged_data['outcome'].replace(to_replace = merged_data['outcome'].unique(), value=[1, 2, 0], inplace=True)
    merged_data['outcome_group'] = merged_data['outcome']
    # -1: nan, 0: Female, 1: Male
    # unique() == [nan 'female' 'male']
    merged_data['sex'].replace(to_replace = merged_data['sex'].unique(), value=[-1, 0, 1], inplace=True)
    # 0: False, 1: True
    # unique() == [False  True]
    merged_data['chronic_disease_binary'].replace(to_replace = merged_data['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)
    merged_data['country']
    
    
    # For test_cases as well
    test_cases['sex'].replace(to_replace = test_cases['sex'].unique(), value=[-1, 0, 1], inplace=True)
    test_cases['chronic_disease_binary'].replace(to_replace = test_cases['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)
    
    print('merged_data: \n', merged_data)
    print('test_cases: \n', test_cases)
    
    # ... add on
    
    # Write out as clean data file
    merged_data.to_csv('./clean_data/merged_data.csv', index=False)
    test_cases.to_csv('./clean_data/test_cases.csv', index=False)
    return
    
if __name__ == '__main__':
    main()