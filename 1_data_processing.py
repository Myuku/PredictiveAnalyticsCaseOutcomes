import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
def main():
    location_data = read_data('./clean_data/location_2021.csv')
    train_cases = read_data('./clean_data/cases_2021_train.csv')
    # test_cases = read_data('./clean_data/cases_2021_test.csv')
    
    # Merge data
    merged_data = pd.merge(train_cases, location_data, how='left', on=['country', 'province'])        # Error note: some province are NaN in location_2021.csv so this is not working currently
    
    # Add Expected Mortality Rate
    # TODO: observed deaths in cases_2021_train (based on location) / Location (country, province) Deaths count   ---> UNSURE! just what I think it is. Please rethink about this. - Alisa
    observed_deaths_data = merged_data[merged_data['outcome'] == 'Deceased'].value_counts(['country', 'province', 'outcome']).reset_index().rename(columns={"count": "observed_deaths"}).drop('outcome', axis=1)
    merged_data = pd.merge(merged_data, observed_deaths_data, how='left', on=['country', 'province']) 
    merged_data['Expected_Mortality_Rate'] = merged_data['observed_deaths'] / merged_data['Deaths']
    
    # Write out as clean data file
    merged_data.to_csv('./clean_data/merged_data.csv', index=False)
    return
    
if __name__ == '__main__':
    main()