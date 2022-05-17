import numpy as np
import csv
from time import time
from abess.linear import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, OrthogonalMatchingPursuitCV
from celer import LassoCV as celerLassoCV
from sklearn.model_selection import train_test_split
import pandas as pd

def metrics(coef, pred, real):
    auc = mean_squared_error(real, pred)

    nnz = len(np.nonzero(coef)[0])

    return np.array([auc, nnz])

M = 20
model_name = "Lm"
method = [
    # "lasso",
    "celer",
    # "omp", 
    "abess",
]
res_output = True
data_output = False

# AUC, NNZ, time
met = np.zeros((len(method), M, 3))
res = np.zeros((len(method), 6))

# read data
# file_name = "popularity"
file_name = "superconduct"

# with open('./superconduct_x.txt', 'r') as f:
#     reader = csv.reader(f)
#     X = [row for row in reader]
# X = np.array(X, dtype = float)
X = pd.read_csv("{0}_x.txt".format(file_name)).to_numpy()

# with open('./superconduct_y.txt', 'r' as f:
#     reader = csv.reader(f)
#     y = [row for row in reader]
# y = np.array(y, dtype = float)
y = pd.read_csv("{0}_y.txt".format(file_name)).to_numpy()
y = np.reshape(y, -1)

print(X.shape)
print(y.shape)

# Test
print('===== Testing '+ model_name + ' =====')
for m in range(M):
    ind = -1
    if (m % 10 == 0):
        print(" --> iter: " + str(m))

    if X.shape[0] <= X.shape[1]:
        trainx, testx, trainy, testy = train_test_split(X, y, test_size = 0.1, random_state = m)
    else:
        trainx, testx, trainy, testy = train_test_split(X, y, test_size = 0.3, random_state = m)    

    if "lasso" in method:
        ind += 1

        t_start = time()
        model = LassoCV(cv = 5, n_jobs = 5)
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> SKL time: " + str(t_end - t_start))
        print("     --> SKL err : " + str(met[ind, m, 0]))
    
    if "celer" in method:
        ind += 1

        t_start = time()
        model = celerLassoCV(cv = 5, n_jobs = 5)
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> CELER time: " + str(t_end - t_start))
        print("     --> CELER err : " + str(met[ind, m, 0]))
    
    ## omp
    if "omp" in method:
        ind += 1

        t_start = time()
        model = OrthogonalMatchingPursuitCV(cv=5, n_jobs=5, max_iter=100)
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> OMP time: " + str(t_end - t_start))

    ## abess
    if "abess" in method:
        ind += 1
        max_supp = np.min([100, trainx.shape[1]])

        t_start = time()
        # model = abessLogistic(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99, thread = 0)
        model = LinearRegression(cv=5, support_size = range(max_supp), thread = 5, important_search=100)
        model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(model.coef_, model.predict(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> ABESS time: " + str(t_end - t_start))
        print("     --> ABESS err : " + str(met[ind, m, 0]))

for ind in range(0, len(method)):
    m = met[ind].mean(axis = 0)
    se = met[ind].std(axis = 0) / np.sqrt(M - 1)
    res[ind] = np.hstack((m, se))

print("===== Results " + model_name + " =====")
print("Method: \n", method)
print("Metrics: \n", res[:, 0:3])
print("Err: \n", res[:, 3:6])

if (res_output):
    np.save(model_name + "_res.npy", res)
    print("Result saved.")

if (data_output):
    np.save(model_name + "_data.npy", met) 
    print("Data saved.")