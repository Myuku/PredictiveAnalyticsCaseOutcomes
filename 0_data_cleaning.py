import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
def main():
    location_data = read_data('./data/location_2021.csv')
    train_cases = read_data('./data/cases_2021_train.csv')
    test_cases = read_data('./data/cases_2021_test.csv')
    
    # Add columns as per needed
    location_data = location_data[['Country_Region', 'Confirmed', 'Deaths', 'Province_State']]
    train_cases = train_cases[['country', 'province', 'outcome']]
    
    # standardize naming convention
    location_data = location_data.rename(columns={'Country_Region': 'country', 'Province_State': 'province'}) 
    location_data.loc[location_data['country'] == 'US', 'country'] = 'United States'
    train_cases['outcome'] = train_cases['outcome'].str.capitalize()
    
    # Fix missing data
    location_data['province'] = location_data['province'].fillna(location_data['country'])
    
    # Write out as clean data file
    location_data.to_csv('./clean_data/location_2021.csv', index=False)
    train_cases.to_csv('./clean_data/cases_2021_train.csv', index=False)
    test_cases.to_csv('./clean_data/cases_2021_test.csv', index=False)
    return
    
if __name__ == '__main__':
    main()