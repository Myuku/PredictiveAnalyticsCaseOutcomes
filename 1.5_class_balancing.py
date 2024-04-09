import pandas as pd
from imblearn.over_sampling import SMOTE

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data

def main():
    train_cases = read_data('./all_data/partB/cleaned_data/train_cases.csv')

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
