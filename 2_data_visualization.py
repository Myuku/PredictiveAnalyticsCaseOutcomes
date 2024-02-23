import sys
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as geo_pd
import numpy as np
import seaborn as sns

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data

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
    total_avaliable_data = df['count'].sum()
    
    merged_world = pd.merge(world, df, left_on=['NAME_CIAWF'], right_on=['country'], how='left')
    
    # Case 1: percentage is based on avaliable data of country / total avaliable data from all countries
    merged_world['avaliable_percentage'] = (merged_world['count'] / total_avaliable_data) * 100         
    merged_world['avaliable_percentage_log'] = np.log(merged_world['avaliable_percentage'])     
    
    # Case 2: percentage is based on: avaliable data of country / country population
    # !!! Problem: values too small 
    # merged_world['avaliable_percentage'] = (np.divide(merged_world['count'], merged_world['POP_EST']) * 100 )  
    
    # Plot Heat Map
    _, ax = plt.subplots(1, 1)
    merged_world.plot(column='avaliable_percentage', ax=ax, cmap="rocket_r", legend=True, legend_kwds={'shrink': 0.6}, vmax=100, vmin=0)       # Only used for proper Legend values
    merged_world.plot(column='avaliable_percentage_log', ax=ax, cmap="rocket_r", legend=False)                             # Using Log Version (for bigger differences)
    # merged_world.plot(column='avaliable_percentage', ax=ax, cmap="rocket_r", legend=False)
    ax.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
    
    # # Plot Bar Plot
    by_continent = merged_world.groupby('CONTINENT').sum('count').reset_index()
    by_continent.loc[by_continent['CONTINENT'] == 'Seven seas (open ocean)', 'CONTINENT'] = 'Seven seas'     # for readibility
    by_continent['log_count'] = np.log(by_continent['count'].replace(0.0, np.nan))
    by_continent.fillna(0, inplace=True)

    y_values_log = by_continent['log_count'].values.tolist()
    
    _, ax1 = plt.subplots(1, 1)
    ax1 = sns.barplot(by_continent, x='CONTINENT', y='log_count', palette=colors_from_values(np.array(y_values_log), "YlOrRd"), hue='CONTINENT', legend=False)
    ax1.set(xlabel='Continents', ylabel='Avaliable Data')
    
    # Add value to the bars
    for p in ax1.patches:
        value = np.round(np.exp(p.get_height()), decimals=0).astype(int) if p.get_height() > 0 else 0
        ax1.annotate(value,
                    xy=(p.get_x()+p.get_width()/2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points')


    plt.tick_params(left = False, labelleft = False)
    plt.xticks(rotation=45)
    plt.show()

    # Save data
    # merged_world.to_csv('./clean_data/world_data.csv')
    # by_continent.to_csv('./clean_data/continent_data.csv')
    
def main():
    location_data = read_data('./clean_data/location_2021.csv')
    train_cases = read_data('./clean_data/cases_2021_train.csv')
    world_data = geo_pd.read_file('./world_map_data/ne_110m_admin_0_countries.shp')
    
    # for testing purposes
    merged_data = pd.concat([train_cases, location_data], ignore_index=True)    # concat on country
    num_avaliable_data_by_country = merged_data[merged_data['outcome'].notna()].value_counts('country').reset_index()
    
    # Plot
    plot_data(num_avaliable_data_by_country, world_data)
    return
    
if __name__ == '__main__':
    main()