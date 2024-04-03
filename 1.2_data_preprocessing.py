import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
def main():
    location_data = read_data('./all_data/partA/clean_data/cleaned_location.csv')
    train_cases = read_data('./all_data/partA/clean_data/cleaned_train_cases.csv')
    
    # Merge data
    merged_data = pd.merge(train_cases, location_data, how='left', on=['country', 'province'])
    # FIXME: Just removed all na values other than in outcome_group.
    #        question says "Find out what is the best strategy for dealing with missing values"
    #        so there prob is a better method
    merged_data = merged_data.dropna(subset = merged_data.columns.difference(['outcome_group']))
    
    # Add Expected Mortality Rate
    # Observed deaths in cases_2021_train (based on location) / Location (country, province) Deaths count
    observed_deaths_data = merged_data[merged_data['outcome'] == 'Deceased']    \
                           .value_counts(['country', 'province', 'outcome']).reset_index()  \
                           .rename(columns={"count": "observed_deaths"}).drop('outcome', axis=1)
    merged_data = pd.merge(merged_data, observed_deaths_data, how='left', on=['country', 'province']) 
    merged_data['Expected_Mortality_Rate'] = merged_data['observed_deaths'] / merged_data['Deaths']
    
    # Write out as clean data file
    merged_data.to_csv('./all_data/partA/clean_data/EMR_data.csv', index=False)
    return
    
if __name__ == '__main__':
    main()