import pandas as pd

def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data
    
    
def main():
    train_cases = read_data('./all_data/partB/clean_data/train_cases.csv')
    
    
    train_cases.to_csv('./all_data/partB/clean_data/balanced_test_cases.csv', index=False)
    return
    
    
if __name__ == '__main__':
    main()