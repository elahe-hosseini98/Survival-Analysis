from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

random_state = 20

def k_best(X_train, X_test , y_train , y_test):
    nof_list=np.arange(1, 20)            
    high_score = 0
    nof = 0           
    score_list =[]
    
    est_cph_tree = GradientBoostingSurvivalAnalysis(
        n_estimators=45, learning_rate=0.1, subsample=0.5,
        max_depth=5, dropout_rate=0.1, random_state=random_state)
    
    for n in nof_list:
        fs = SelectKBest(f_classif, k=n)
        relief = Pipeline([('fs', fs), ('m', est_cph_tree)])
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

def survival_gradient_boosted(X_train, X_test, y_train, y_test):
    X_train, X_test = k_best(X_train, X_test, y_train, y_test)
    print('GB X_train shape: ',X_train.shape)
    
    n_estimators = [i * 5 for i in range(1, 21)]
    estimators = {
        "no regularization": GradientBoostingSurvivalAnalysis(
            learning_rate=1.0, max_depth=5, random_state=random_state
        ),
        "learning rate": GradientBoostingSurvivalAnalysis(
            learning_rate=0.5, max_depth=5, random_state=random_state
        ),
        "dropout": GradientBoostingSurvivalAnalysis(
            learning_rate=1.0, dropout_rate=0.1, max_depth=5, 
            random_state=random_state
        ),
        "subsample": GradientBoostingSurvivalAnalysis(
            learning_rate=1.0, subsample=0.5, max_depth=5, random_state=random_state
        ),
    }
    
    scores_reg = {k: [] for k in estimators.keys()}
    for n in n_estimators:
        for name, est in estimators.items():
            est.set_params(n_estimators=n)
            est.fit(X_train, y_train)
            cindex = est.score(X_test, y_test)
            scores_reg[name].append(cindex)
    
    scores_reg = pd.DataFrame(scores_reg, index=n_estimators)
    ax = scores_reg.plot(xlabel="n_estimators", ylabel="concordance index")
    ax.grid(True)
    plt.title('Gradient Boosted')
    plt.show()
    
    scores_cph_tree = {}

    est_cph_tree = GradientBoostingSurvivalAnalysis(
        learning_rate=0.1, subsample=0.5,
        max_depth=5, dropout_rate=0.1, random_state=random_state
    )
    for i in range(1, 21):
        n_estimators = i * 5
        est_cph_tree.set_params(n_estimators=n_estimators)
        est_cph_tree.fit(X_train, y_train)
        scores_cph_tree[n_estimators] = est_cph_tree.score(X_test, y_test)

    best_estimators = max(scores_cph_tree, key=scores_cph_tree.get)
    print('GB best estimators count =', best_estimators)
    
    est_cph_tree = GradientBoostingSurvivalAnalysis(
        n_estimators=best_estimators, learning_rate=0.1, subsample=0.5,
        max_depth=5, dropout_rate=0.1, random_state=random_state)
    
    est_cph_tree.fit(X_train, y_train)
    c_index = est_cph_tree.score(X_test, y_test)
    return c_index

