# %%
import numpy as np
import pandas as pd
from time import time
from abess.linear import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from celer import LogisticRegression as celerLogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %% read data
X = pd.read_csv('chin_x.txt', header=0)
X = X.to_numpy()
y = pd.read_csv('chin_y.txt', header=0)
y = np.array(y, dtype = float)
y = np.reshape(y, -1)
print("sample size: {0}, dimension: {1}".format(X.shape[0], X.shape[1]))

# %% evaluation

def metrics(coef, pred, real):
    auc = roc_auc_score(real, pred)

    nnz = len(np.nonzero(coef)[0])

    return np.array([auc, nnz])

M = 1
model_name = "Logistic"
method = [
    "lasso",
    "celer", 
    "abess",
]
res_output = True
data_output = False
verbose = True

# AUC, NNZ, time
met = np.zeros((len(method), M, 3))
res = np.zeros((len(method), 6))

# %% Test
print('===== Testing '+ model_name + ' =====')
for m in range(M):
    ind = -1
    print(" --> Replication: " + str(m+1))

    trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.1, random_state=m)

    # method 1:
    # alphas, t1, t2, t3 = celer_path(trainx, 2 * trainy - 1, pb="logreg")
    # method 2 (https://mathurinm.github.io/celer/auto_examples/plot_finance_path.html#sphx-glr-auto-examples-plot-finance-path-py):
    alpha_max = np.max(np.abs(trainx.T.dot(trainy))) / trainx.shape[0]
    n_alphas = 100
    alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)

    ## lasso
    if "lasso" in method:
        ind += 1

        t_start = time()
        model = LogisticRegressionCV(Cs=alphas, penalty="l1", 
                                     solver = "liblinear", cv=5, n_jobs=5, random_state=0)
        fit = model.fit(trainx, trainy)
        t_end = time()
        best_lasso_C = fit.C_[0]

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict_proba(testx)[:, 1].flatten(), testy)
        met[ind, m, 2] = t_end - t_start

        if verbose:
            print("     --> SKL time: " + str(t_end - t_start))
            print("     --> SKL AUC : " + str(met[ind, m, 0]))
            print("     --> SKL NNZ : " + str(met[ind, m, 1]))

    ## celer
    if "celer" in method:
        ind += 1
      
        tune_celer = False
        if tune_celer:
            parameters = {'C': alphas}
            t_start = time()
            model = celerLogisticRegressionCV()
            model = GridSearchCV(model, parameters, n_jobs=-1, cv=5)
            fit = model.fit(trainx, trainy)
            t_end = time()

            met[ind, m, 0:2] = metrics(fit.best_estimator_.coef_, fit.predict_proba(testx)[:, 1].flatten(), testy)
        else:
            t_start = time()
            # ConvergenceWarning occurs, so increase `tol` and `max_iter`
            model = celerLogisticRegressionCV(C=best_lasso_C)
            fit = model.fit(trainx, trainy)
            t_end = time()

            met[ind, m, 0:2] = metrics(fit.coef_, fit.predict_proba(testx)[:, 1].flatten(), testy)

        met[ind, m, 2] = t_end - t_start

        if verbose:
            print("     --> Celer time: " + str(t_end - t_start))
            print("     --> Celer AUC : " + str(met[ind, m, 0]))
            print("     --> Celer NNZ : " + str(met[ind, m, 1]))
    
    ## abess
    if "abess" in method:
        ind += 1

        t_start = time()
        # model = abessLogistic(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99, thread = 0)
        model = LogisticRegression(cv=5, support_size = range(100), thread=5, 
                                   approximate_Newton = True, primary_model_fit_epsilon=1e-6)
        model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(model.coef_, model.predict_proba(testx)[:, 1].flatten(), testy)
        met[ind, m, 2] = t_end - t_start

        if verbose:
            print("     --> ABESS time: " + str(t_end - t_start))
            print("     --> ABESS AUC : " + str(met[ind, m, 0]))
            print("     --> ABESS NNZ : " + str(met[ind, m, 1]))

for ind in range(0, len(method)):
    m = met[ind].mean(axis = 0)
    se = met[ind].std(axis = 0) / np.sqrt(M - 1)
    res[ind] = np.hstack((m, se))
    res = np.around(res, decimals=2)

print("===== Results " + model_name + " =====")
print("Method: \n", method)
print("Metrics (AUC, NNZ, Runtime): \n", res[:, 0:3])

file_name = "chin"
if (res_output):
    np.save("{}_{}_res.npy".format(model_name, file_name), res)
    print("Result saved.")

