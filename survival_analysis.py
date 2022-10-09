import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from random_Survival_forests import random_survival_forests
from survival_gradient_boosted import survival_gradient_boosted
from survival_SVM import fast_Kernel_Survival_SVM

random_state = 20

def splitting_from_each(df, unique_centers, test_size=.2):
    df = df.sample(frac=1).reset_index(drop=True)
    
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    scaler = MinMaxScaler()
    
    for center in unique_centers:
        center_train_df = df.loc[df['Center'] == center]
        center_train_size = int(center_train_df.shape[0]*(1-test_size))
        train_df = pd.concat([train_df, center_train_df[:center_train_size]], sort=False)
        test_df = pd.concat([test_df, df.loc[df['Center'] == center][center_train_size:]], sort=False)       

    X_train = train_df.iloc[:, 5:]
    X_train = scaler.fit_transform(X_train)
    y_train = train_df[['censor', 'time']]
    y_train = y_train.to_records(index=False, column_dtypes={'censor': 'bool'})
    
    X_test = test_df.iloc[:, 5:]
    X_test = scaler.fit_transform(X_test)
    y_test = test_df[['censor', 'time']]
    y_test = y_test.to_records(index=False, column_dtypes={'censor': 'bool'})
    
    print('train size =', train_df.shape[0], '\ntest size =', test_df.shape[0])

    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
    return X_train, X_test, y_train, y_test

def splitting_one_center_test(df, test_center):
    train_df = df.loc[df['Center'] != test_center]
    scaler = MinMaxScaler()
    
    X_train = train_df.iloc[:, 5:]
    X_train = scaler.fit_transform(X_train)   
    
    y_train = train_df[['censor', 'time']]
    y_train = y_train.to_records(index=False, column_dtypes={'censor': 'bool'}) # convert y_train df to recunstructed np array
    
    # test set
    test_df = df.loc[df['Center'] == test_center]
    
    X_test = test_df.iloc[:, 5:]
    X_test = scaler.fit_transform(X_test)   
    
    y_test = test_df[['censor', 'time']]
    y_test = y_test.to_records(index=False, column_dtypes={'censor': 'bool'}) # convert y_test df to recunstructed np array
    
    print('train size =', train_df.shape[0], '\ntest size =', test_df.shape[0])

    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
    return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":  
    dataset_path = r'Before_PFS.xlsx'
    df = pd.read_excel(dataset_path)
    
    print('dataset size:', df.shape[0])
    
    unique_centers = pd.unique(df['Center'])
    print('unique centers name:', unique_centers)
    
    result_df = pd.DataFrame(columns=['Test_Center', 'random_survival_forests', 'survival_gradient_boosted', 'fast_Kernel_Survival_SVM'])

    # 20% From each spliting
    X_train, X_test, y_train, y_test = splitting_from_each(df, unique_centers)
    RSF_C_index = random_survival_forests(X_train, X_test, y_train, y_test)
    GB_C_index = survival_gradient_boosted(X_train, X_test, y_train, y_test)
    FSVM_C_index = fast_Kernel_Survival_SVM(X_train, X_test, y_train, y_test)
    result_data = [{'Test_Center':'20%_from_each', 'random_survival_forests':RSF_C_index,
                    'survival_gradient_boosted':GB_C_index, 'fast_Kernel_Survival_SVM':FSVM_C_index}]
    result_df = result_df.append(result_data, ignore_index=True)
    
    # One center out spliting
    for i, center_name in enumerate(unique_centers):
        
        X_train, X_test, y_train, y_test = splitting_one_center_test(df, center_name)
    
        RSF_C_index = random_survival_forests(X_train, X_test, y_train, y_test)
        GB_C_index = survival_gradient_boosted(X_train, X_test, y_train, y_test)
        FSVM_C_index = fast_Kernel_Survival_SVM(X_train, X_test, y_train, y_test)
        
        result_data = [{'Test_Center':center_name, 'random_survival_forests':RSF_C_index,
                        'survival_gradient_boosted':GB_C_index, 'fast_Kernel_Survival_SVM':FSVM_C_index}]
        
        result_df = result_df.append(result_data, ignore_index=True)
    
    result_df.to_excel('result.xlsx', index=False)
    
    
    
    
    
