import pandas as pd
import matplotlib.pyplot as plt
import geopandas as geo_pd
import numpy as np
import seaborn as sns

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
    
    cases.loc['outcome'] = cases['outcome'].str.capitalize()
    cases.loc[cases['outcome'].isin(hospitalized), 'outcome'] = 'Hospitalized'
    cases.loc[cases['outcome'].isin(non_hospitalized), 'outcome'] = 'Non-Hospitalized'
    cases.loc[cases['outcome'].isin(deceased), 'outcome'] = 'Deceased'
    # Handle cases where there is more cases than what is known. Can't use 'Other' due to project req.
    cases.loc[~cases['outcome'].isin(['Hospitalized', 'Non-Hospitalized', 'Deceased']), 'outcome'] = 'Non-Hospitalized'
    return

# Colour Palette with Indices for Seaborn
# Reference: https://stackoverflow.com/questions/36271302/changing-color-scale-in-seaborn-bar-plot
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return list(np.array(palette).take(indices, axis=0))

'''
    Possible relavant columns in world with their examples are: 
        POSTAL = 'US', 
        NAME_EN = 'United States of America', 
        NAME_CIAWF = 'United States'
        CONTINENT = 'North America'
        POP_EST => population estimate based on POP_YEAR
'''
def plot_data(df: pd.DataFrame, world: geo_pd.GeoDataFrame):
    total_available_data = df['count'].sum()
    
    merged_world = pd.merge(world, df, left_on=['NAME_CIAWF'], right_on=['country'], how='left')
    
    # Case 1: percentage is based on avaliable data of country / total avaliable data from all countries
    merged_world['available_percentage'] = (merged_world['count'] / total_available_data) * 100   
    merged_world['available_percentage_log'] = np.log(merged_world['available_percentage'])  
      
    
    # Case 2: percentage is based on: avaliable data of country / country population
    # !!! Problem: values too small 
    # merged_world['available_percentage'] = (np.divide(merged_world['count'], merged_world['POP_EST']) * 100 )  
    # merged_world['available_percentage_log'] = np.log(merged_world['available_percentage'])  
    
    # Plot Heat Map
    _, ax = plt.subplots(1, 1)
    merged_world.plot(column='available_percentage', ax=ax, cmap="rocket_r", legend=True, legend_kwds={'shrink': 0.6}, vmax=100, vmin=0)       # Only used for proper Legend values
    merged_world.plot(column='available_percentage_log', ax=ax, cmap="rocket_r", legend=False)                             # Using Log Version (for bigger differences)
    # merged_world.plot(column='available_percentage', ax=ax, cmap="rocket_r", legend=False)
    ax.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
    
    # # Plot Bar Plot
    by_continent = merged_world.groupby('CONTINENT').sum('count').reset_index()
    by_continent.loc[by_continent['CONTINENT'] == 'Seven seas (open ocean)', 'CONTINENT'] = 'Seven seas'     # for readibility
    by_continent['log_count'] = np.log(by_continent['count'].replace(0.0, np.nan))
    by_continent.fillna(0, inplace=True)

    y_values_log = by_continent['log_count'].values.tolist()
    
    _, ax1 = plt.subplots(1, 1)
    ax1 = sns.barplot(by_continent, x = 'CONTINENT', y = 'log_count', 
                      palette = colors_from_values(np.array(y_values_log), "YlOrRd"), hue = 'CONTINENT', legend = False)
    ax1.set(xlabel='Continents', ylabel='Available Data')
    
    # Add value to the bars
    for p in ax1.patches:
        # Revert the log to use correct numbers when showcasing values
        value = np.round(np.exp(p.get_height()), decimals=0).astype(int) if p.get_height() > 0 else 0
        ax1.annotate(value,
                    xy = (p.get_x()+p.get_width()/2., p.get_height()),
                    ha = 'center',
                    va = 'center',
                    xytext = (0, 10),
                    textcoords = 'offset points')


    plt.tick_params(left = False, labelleft = False)
    plt.xticks(rotation = 45)
    plt.show()

    # Save data
    # merged_world.to_csv('./clean_data/world_data.csv')
    # by_continent.to_csv('./clean_data/continent_data.csv')
    
def clean_dataset(location_data: pd.DataFrame, train_cases: pd.DataFrame):
    location_data = location_data[['Country_Region', 'Confirmed', 'Deaths', 'Province_State']]
    # age,sex,province,country,latitude,longitude,date_confirmation,additional_information,source,chronic_disease_binary
    train_cases = train_cases[['age', 'sex', 'country', 'province', 'chronic_disease_binary', 'outcome', 'outcome_group']]
    # standardize naming convention
    clean_cases(train_cases)
    location_data = location_data.rename(columns={'Country_Region': 'country', 'Province_State': 'province'}) 
    location_data.loc[location_data['country'] == 'US', 'country'] = 'United States'
    
    # Fix missing data
    location_data['province'] = location_data['province'].fillna(location_data['country'])
    
    location_data.to_csv('./clean_data/cleaned_location.csv', index = False)
    train_cases.to_csv('./clean_data/cleaned_train_cases.csv', index = False)
    
    return train_cases, location_data
    
def main():
    location_data = read_data('./data/location_2021.csv')
    train_cases = read_data('./data/cases_2021_train.csv')
    world_data = geo_pd.read_file('./world_map_data/ne_110m_admin_0_countries.shp')
    
    location_data, train_cases = clean_dataset(location_data, train_cases)
    
    # Merge data set
    merged_data = pd.concat([train_cases, location_data], ignore_index = True)    # concat on country
    num_avaliable_data_by_country = merged_data[merged_data['outcome'].notna()] \
                                    .value_counts('country').reset_index()
    # Plot
    plot_data(num_avaliable_data_by_country, world_data)
    return
    
if __name__ == '__main__':
    main()