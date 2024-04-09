import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data

def reduce_outliers(df, column_names, remove=False):
    outlier_indices = []
    for column in column_names:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        column_outliers = df.index[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_indices.extend(column_outliers)
    
    outlier_indices = list(set(outlier_indices))  # Remove duplicates
    if remove:
        return df.drop(index=outlier_indices)
    else:
        return df.loc[outlier_indices]
    

def visualize_outliers(df, column_names):
    for column in column_names:
        plt.figure(figsize=(10, 6))
        df.boxplot(column=[column])
        plt.title(f"Box plot for {column}")
        plt.show()
    
def main():
    train_cases = read_data('./all_data/partB/cleaned_data/train_cases.csv')
    print(train_cases['Case_Fatality_Ratio'].value_counts())
    
    column_names_to_check = ['Case_Fatality_Ratio'] 
    
    # Visualizing outliers before removal
    print("Before outlier removal:")
    visualize_outliers(train_cases, column_names_to_check)
    
    # Removing outliers
    train_cases_no_outliers = reduce_outliers(train_cases, column_names_to_check, remove=True)
    print(train_cases_no_outliers['Case_Fatality_Ratio'].value_counts())
    # Visualizing data after outlier removal
    print("After outlier removal:")
    visualize_outliers(train_cases_no_outliers, column_names_to_check)
    
    # We discover that removing the outliers make the prediction worst
    
    # For train_cases
    # Display original outcome distribution in the training set
    train_cases_outcome = train_cases['outcome_group'].value_counts()
    print("Original outcome distribution in training data:\n", train_cases_outcome)

    # Separating features and target variable in the training dataset
    X_train = train_cases.drop('outcome_group', axis=1)
    y_train = train_cases['outcome_group']
    

    # Applying SMOTE to balance the classes in the training dataset
    # random_state can be change between 0~42 (doesn't matter)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    balanced_train_cases = pd.concat([X_train_resampled, y_train_resampled], axis=1)

    print("Resampled outcome distribution in training data:\n", balanced_train_cases['outcome_group'].value_counts())
    balanced_train_cases.to_csv('./all_data/partB/balanced_data/train_cases.csv', index=False)



if __name__ == '__main__':
    main()
