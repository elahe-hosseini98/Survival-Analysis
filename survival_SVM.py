from sksurv.svm import FastSurvivalSVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

random_state = 20

def k_best(X_train, X_test , y_train , y_test):
    nof_list=np.arange(1, 20)            
    high_score = 0
    nof = 0           
    score_list =[]
    
    fsvm = FastSurvivalSVM(rank_ratio=0.1, max_iter=1000, tol=1e-5,
                                    random_state=random_state)
    
    for n in nof_list:
        fs = SelectKBest(f_classif, k=n)
        relief = Pipeline([('fs', fs), ('m', fsvm)])
        relief.fit(X_train, y_train)
        score = relief.score(X_test, y_test)
        score_list.append(score)
        if(score > high_score):
            high_score = score
            nof = n
    
    selector = SelectKBest(f_classif, k=nof)
    selector.fit(X_train, y_train)
    cols = selector.get_support(indices=True)
    
    X_train = X_train[:][:, cols]
    X_test = X_test[:][:, cols]
    return X_train, X_test

def fast_Kernel_Survival_SVM(X_train, X_test , y_train , y_test):
    X_train, X_test = k_best(X_train, X_test, y_train, y_test)
    print('Fast SVM X_train shape: ',X_train.shape)

    scores_fsvm_ls = {}
    est_fsvm_ls = FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=random_state)
    
    for i in range(0, 10):
        rank_ratio = i * 0.1
        est_fsvm_ls.set_params(rank_ratio=rank_ratio)
        est_fsvm_ls.fit(X_train, y_train)
        score = est_fsvm_ls.score(X_test, y_test)
        scores_fsvm_ls[rank_ratio] = score
   
    x, y = zip(*scores_fsvm_ls.items())
    plt.plot(x, y)
    plt.xlabel("rank_ratio")
    plt.ylabel("concordance index")
    plt.grid(True)
    plt.title('Fast SVM')
    plt.show()
    
    best_rank_ratio = max(scores_fsvm_ls, key=scores_fsvm_ls.get)
    print('Fast SVM best rank_ratio =', best_rank_ratio)
    
    fsvm = FastSurvivalSVM(rank_ratio=best_rank_ratio, max_iter=1000, tol=1e-5,
                                    random_state=random_state)
    fsvm.fit(X_train, y_train)
    c_index = fsvm.score(X_test, y_test)
    return c_index