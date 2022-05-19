import numpy as np
import csv
from time import time
from abess.linear import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from celer import LogisticRegression as celerLogisticRegressionCV
from celer import celer_path
from sklearn.model_selection import train_test_split, GridSearchCV
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def metrics(coef, pred, real):
    auc = roc_auc_score(real, pred)

    nnz = len(np.nonzero(coef)[0])

    return np.array([auc, nnz])

M = 20
model_name = "Logistic"
method = [
    "lasso",
    # "celer", 
    "abess",
]
res_output = True
data_output = False
verbose = True

# AUC, NNZ, time
met = np.zeros((len(method), M, 3))
res = np.zeros((len(method), 6))

# read data
with open('./chowdary_x.txt', 'r') as f:
    reader = csv.reader(f)
    X = [row for row in reader]
X = np.array(X, dtype = float)

with open('./chowdary_y.txt', 'r') as f:
    reader = csv.reader(f)
    y = [row for row in reader]
y = np.array(y, dtype = float)
y = np.reshape(y, -1)


# Test
print('===== Testing '+ model_name + ' =====')
for m in range(M):
    ind = -1
    print(" --> Replication: " + str(m+1))

    trainx, testx, trainy, testy = train_test_split(X, y, test_size = 0.1, random_state = m)

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
        model = LogisticRegressionCV(Cs=alphas, penalty="l1", solver = "liblinear", cv=5, n_jobs=5)
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict_proba(testx)[:, 1].flatten(), testy)
        met[ind, m, 2] = t_end - t_start

        if verbose:
            print("     --> SKL time: " + str(t_end - t_start))

    ## celer
    if "celer" in method:
        ind += 1
      
        parameters = {'C': alphas}
        model = celerLogisticRegressionCV()
        model = GridSearchCV(model, parameters, n_jobs=-1, cv=5)
        t_start = time()
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.best_estimator_.coef_, fit.predict_proba(testx)[:, 1].flatten(), testy)
        met[ind, m, 2] = t_end - t_start

        if verbose:
            print("     --> Celer time: " + str(t_end - t_start))
    
    ## abess
    if "abess" in method:
        ind += 1

        t_start = time()
        # model = abessLogistic(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99, thread = 0)
        model = LogisticRegression(cv=5, support_size = range(100), thread = 5, 
                                   approximate_Newton = True, primary_model_fit_epsilon=1e-6)
        model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(model.coef_, model.predict_proba(testx), testy)
        met[ind, m, 2] = t_end - t_start

        if verbose:
            print("     --> ABESS time: " + str(t_end - t_start))

for ind in range(0, len(method)):
    m = met[ind].mean(axis = 0)
    se = met[ind].std(axis = 0) / np.sqrt(M - 1)
    res[ind] = np.hstack((m, se))
    res = np.around(res, decimals=2)

print("===== Results " + model_name + " =====")
print("Method: \n", method)
print("Metrics (AUC, NNZ, Runtime): \n", res[:, 0:3])

if (res_output):
    np.save(model_name + "_res.npy", res)
    print("Result saved.")
