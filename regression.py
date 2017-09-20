## import dependnent libraries
import sklearn.linear_model
import numpy as np
from scipy import stats


## one-shot ridge regression 

def one_shot_regression(X, y, lamb):
    clf=sklearn.linear_model.Ridge(alpha=lamb,fit_intercept=True,normalize=False,copy_X=True,max_iter=None,tol=0.001,solver='auto',random_state=None)
    result=clf.fit(X,y)
    beta_vector=np.insert(result.coef_,0,result.intercept_)
    return beta_vector

def y_estimate(biased_X, beta_vector):
    return np.dot(beta_vector, np.matrix.transpose(biased_X))


## rSquared calculation
def r_square(biased_X, y, beta_vector):
    SSE = np.sum(np.square(y-y_estimate(biased_X, beta_vector)))
    SST = np.sum(np.square(y-np.mean(y)))
    return (1 - (SSE/SST))


## t value calculation
def t_value(biased_X, y, beta_vector):
    MSE = np.sum(np.square(y-y_estimate(biased_X, beta_vector)))/(len(y) - len(beta_vector))
    var_beta = MSE*(np.linalg.inv(np.dot(biased_X.T,biased_X)).diagonal())
    sd_beta = np.sqrt(var_beta)
    ts_beta = beta_vector/sd_beta
    return ts_beta
 
## t to p value transformation(two tail)
def t_to_p(dof,ts_beta):
    # dof here should equal to the len(y) - len(beta_vector)
    p_values =[2*(1-stats.t.cdf(np.abs(t),dof)) for t in ts_beta]
    return p_values


