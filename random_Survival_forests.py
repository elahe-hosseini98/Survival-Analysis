from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

random_state = 20

def k_best(X_train, X_test , y_train , y_test):
    nof_list=np.arange(1, 20)            
    high_score = 0
    nof = 0           
    score_list =[]
    
    rsf = RandomSurvivalForest(n_estimators=30,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=random_state)
    
    for n in nof_list:
        fs = SelectKBest(f_classif, k=n)
        relief = Pipeline([('fs', fs), ('m', rsf)])
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


def random_survival_forests(X_train, X_test, y_train, y_test):
    X_train, X_test = k_best(X_train, X_test, y_train, y_test)
    print('RF X_train shape: ',X_train.shape)
    
    scores_rsf_ls = {}
    est_rsf_ls = RandomSurvivalForest(
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=random_state)
    for i in range(1, 21):
        n_estimators = i * 5
        est_rsf_ls.set_params(n_estimators=n_estimators)
        est_rsf_ls.fit(X_train, y_train)
        score = est_rsf_ls.score(X_test, y_test)
        scores_rsf_ls[n_estimators] = score
   
    x, y = zip(*scores_rsf_ls.items())
    plt.plot(x, y)
    plt.xlabel("n_estimator")
    plt.ylabel("concordance index")
    plt.grid(True)
    plt.title('Random Forests')
    plt.show()
    
    best_estimators = max(scores_rsf_ls, key=scores_rsf_ls.get)
    print('RF best estimators count =', best_estimators)
    
    rsf = RandomSurvivalForest(n_estimators=best_estimators,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=random_state)
    rsf.fit(X_train, y_train)
    c_index = rsf.score(X_test, y_test)
    return c_index
    

