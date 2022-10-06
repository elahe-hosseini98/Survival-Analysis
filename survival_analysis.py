import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from random_Survival_forests import random_survival_forests
from survival_gradient_boosted import survival_gradient_boosted
from survival_SVM import fast_Kernel_Survival_SVM

random_state = 20

def spliting(df, test_center):
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
    
    result_df = pd.DataFrame(columns=['Test_Center', 'RSF_C_index', 'GB_C_index'])
    
    for i, center_name in enumerate(unique_centers):
        
        X_train, X_test, y_train, y_test = spliting(df, unique_centers[i])
    
        RSF_C_index = random_survival_forests(X_train, X_test, y_train, y_test)
        GB_C_index = survival_gradient_boosted(X_train, X_test, y_train, y_test)
        FSVM_C_index = fast_Kernel_Survival_SVM(X_train, X_test, y_train, y_test)
        
        result_data = [{'Test_Center':center_name, 'RSF_C_index':RSF_C_index,
                        'GB_C_index':GB_C_index, 'FSVM_C_index':FSVM_C_index}]
        
        result_df = result_df.append(result_data, ignore_index=True)
    
    result_df.to_excel('result.xlsx', index=False)
    
    
    
    
    
