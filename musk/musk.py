import numpy as np
import pandas as pd
import csv
from time import time
from abess.linear import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from celer import LogisticRegression as celerLogisticRegressionCV
from celer import celer_path
from sklearn.model_selection import train_test_split, GridSearchCV

def metrics(coef, pred, real):
    auc = roc_auc_score(real, pred)
    nnz = len(np.nonzero(coef)[0])
    return np.array([auc, nnz])

M = 3
model_name = "Logistic"
method = [
    "lasso",
    "celer", 
    "abess",
]
res_output = True
data_output = False

# AUC, NNZ, time
met = np.zeros((len(method), M, 3))
res = np.zeros((len(method), 6))

# read data
file_name = "musk"
# file_name = "mushroom.csv"; label = 'V1'
# file_name = "adult-processed.csv"; label = 'income'
# file_name = "dota.csv"; label = 'V1'
# file_name = "phpV5QYya.csv"; label = 'Result'
# file_name = "connect-binary.data"; label = 'win'  ## LASSO > abess
if file_name in ["musk"]:
    with open('./{0}_x.txt'.format(file_name), 'r') as f:
        reader = csv.reader(f)
        X = [row for row in reader]
    X = np.array(X, dtype = float)

    with open('./{0}_y.txt'.format(file_name), 'r') as f:
        reader = csv.reader(f)
        y = [row for row in reader]
    y = np.array(y, dtype = float)
    y = np.reshape(y, -1)
else:
    raw_file = pd.read_csv(file_name)
    y = raw_file[label]
    if y.dtype != 'int64':
        y = pd.factorize(y)[0]
    # y = y.to_numpy()
    y = np.array(y, dtype = float)
    y = y.reshape(-1, 1)
    X = raw_file.copy()
    X = X.drop(label, axis=1)
    X = pd.get_dummies(X, drop_first=True)
    X = X.to_numpy()
    X = np.array(X, dtype = float)
    pass
print("sample size: {0}, dimension: {1}".format(X.shape[0], X.shape[1]))



# Test
print('===== Testing '+ model_name + ' =====')
for m in range(M):
    ind = -1
    if (m % 10 == 0):
        print(" --> iter: " + str(m))

    trainx, testx, trainy, testy = train_test_split(X, y, test_size = 0.1, random_state = m)

    # method 1:
    # alphas, t1, t2, t3 = celer_path(trainx, 2 * trainy - 1, pb="logreg")
    # method 2 (https://mathurinm.github.io/celer/auto_examples/plot_finance_path.html#sphx-glr-auto-examples-plot-finance-path-py):
    alpha_max = np.max(np.abs(trainx.T.dot(trainy))) / trainx.shape[0]
    n_alphas = X.shape[1]
    alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)

    ## lasso
    if "lasso" in method:
        ind += 1

        t_start = time()
        model = LogisticRegressionCV(Cs = alphas, penalty = "l1", solver = "liblinear", cv = 5, n_jobs = 5)
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict_proba(testx)[:, 1].flatten(), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> SKL time: " + str(t_end - t_start))
        print("     --> SKL AUC: {0}".format(met[ind, m, 0]))
        print("     --> SKL NNZ: {0}".format(met[ind, m, 1]))

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
        print("     --> Celer time: " + str(t_end - t_start))
        print("     --> Celer AUC: {0}".format(met[ind, m, 0]))
        print("     --> Celer NNZ: {0}".format(met[ind, m, 1]))
    
    ## abess
    if "abess" in method:
        ind += 1

        # max_supp = np.min([100, trainx.shape[1]])
        max_supp = trainx.shape[1]
        t_start = time()
        # model = abessLogistic(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99, thread = 0)
        model = LogisticRegression(cv=5, support_size = range(max_supp), thread=5, important_search=100,
                                   approximate_Newton = True, primary_model_fit_epsilon=1e-6)
        model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(model.coef_, model.predict_proba(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> ABESS time: " + str(t_end - t_start))
        print("     --> ABESS AUC: {0}".format(met[ind, m, 0]))
        print("     --> ABESS NNZ: {0}".format(met[ind, m, 1]))

for ind in range(0, len(method)):
    m = met[ind].mean(axis = 0)
    se = met[ind].std(axis = 0) / np.sqrt(M - 1)
    res[ind] = np.hstack((m, se))

print("===== Results " + model_name + " =====")
print("Method: \n", method)
print("Metrics: \n", res[:, 0:3])
print("Err: \n", res[:, 3:6])

if (res_output):
    np.save(model_name + file_name + "_res.npy", res)
    print("Result saved.")

if (data_output):
    np.save(model_name + file_name + "_raw_res.npy", met) 
    print("Data saved.")