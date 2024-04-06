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
    
    # TODO: Combine Country and Province and remove their individual columns
    train_cases['combined_key'] = train_cases['country'] + ', ' + train_cases['province']
    test_cases['combined_key'] = test_cases['country'] + ', ' + test_cases['province']
    train_cases.drop('country', axis=1, inplace=True)
    test_cases.drop('country', axis=1, inplace=True)
    train_cases.drop('province', axis=1, inplace=True)
    test_cases.drop('province', axis=1, inplace=True)
    

    # TODO: Add for country and province
    # Factorize 'country' and 'province' to encode them as integers
    # Country: {'Algeria': 0, 'Angola': 1, 'Australia': 2, 'Bahamas': 3, 'Brazil': 4, 'Burkina Faso': 5, 'Cabo Verde': 6, 'Cameroon': 7, 'Canada': 8, 
    #           'Central African Republic': 9, 'China': 10, 'Cuba': 11, 'Eswatini': 12, 'Ethiopia': 13, 'France': 14, 'Gabon': 15, 'Gambia': 16, 'Germany': 17, 
    #           'Ghana': 18, 'Guinea': 19, 'Guyana': 20, 'India': 21, 'Italy': 22, 'Malaysia': 23, 'Mozambique': 24, 'Nepal': 25, 'Niger': 26, 'Philippines': 27, 
    #           'Romania': 28, 'San Marino': 29, 'Singapore': 30, 'South Korea': 31, 'Sudan': 32, 'Taiwan': 33, 'Tanzania': 34, 'Togo': 35, 'Vietnam': 36, 'Zimbabwe': 37}
    # Province: {'Andhra Pradesh': 0, 'Assam': 1, 'Bayern': 2, 'Bihar': 3, 'British Columbia': 4, 'Chandigarh': 5, 'Chhattisgarh': 6, 'Delhi': 7, 'Gansu': 8, 
    #           'Goa': 9, 'Gujarat': 10, 'Haryana': 11, 'Himachal Pradesh': 12, 'Hong Kong': 13, 'Hubei': 14, 'Jammu and Kashmir': 15, 'Jharkhand': 16, 'Johor': 17, 
    #           'Karnataka': 18, 'Kerala': 19, 'Ladakh': 20, 'Lombardia': 21, 'Madhya Pradesh': 22, 'Maharashtra': 23, 'Manipur': 24, 'Meghalaya': 25, 'Mizoram': 26, 
    #           'Odisha': 27, 'Ontario': 28, 'Puducherry': 29, 'Punjab': 30, 'Queensland': 31, 'Rajasthan': 32, 'Sao Paulo': 33, 'Shaanxi': 34, 'South Australia': 35, 
    #           'Tamil Nadu': 36, 'Telangana': 37, 'Tripura': 38, 'Uttar Pradesh': 39, 'Uttarakhand': 40, 'Veneto': 41, 'West Bengal': 42, 'Zhejiang': 43, nan: 44}
    le = LabelEncoder()
    # train_cases['country'] = le.fit_transform(train_cases['country'])
    # train_cases['province'] = le.fit_transform(train_cases['province'])
    train_cases['combined_key'] = le.fit_transform(train_cases['combined_key'])
    # print(dict(zip(le.classes_, le.transform(le.classes_))))

    # For test_cases as well
    test_cases['sex'].replace(to_replace = test_cases['sex'].unique(), value=[0, 1], inplace=True)
    test_cases['chronic_disease_binary'].replace(to_replace = test_cases['chronic_disease_binary'].unique(),
                                                  value=[0, 1], inplace=True)
    # test_cases['country'] = le.fit_transform(test_cases['country'])
    # test_cases['province'] = le.fit_transform(test_cases['province'])
    test_cases['combined_key'] = le.fit_transform(test_cases['combined_key'])
    

    # TODO: These features do not seem to impact the performance (at least in XGBoost)
    train_cases.drop('Confirmed', axis=1, inplace=True)
    test_cases.drop('Confirmed', axis=1, inplace=True)
    train_cases.drop('Active', axis=1, inplace=True)
    test_cases.drop('Active', axis=1, inplace=True)
    train_cases.drop('Incident_Rate', axis=1, inplace=True)
    test_cases.drop('Incident_Rate', axis=1, inplace=True)
    train_cases.drop('Recovered', axis=1, inplace=True)
    test_cases.drop('Recovered', axis=1, inplace=True)
    train_cases.drop('Deaths', axis=1, inplace=True)
    test_cases.drop('Deaths', axis=1, inplace=True)
    
    print('train_cases: \n', train_cases)
    print('test_cases: \n', test_cases)
    
    # Write out as clean data file
    train_cases.to_csv('./all_data/partB/cleaned_data/train_cases.csv', index = False)
    test_cases.to_csv('./all_data/partB/cleaned_data/test_cases.csv', index = False)
    return
    
if __name__ == '__main__':
    main()