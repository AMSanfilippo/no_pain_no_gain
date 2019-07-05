import numpy as np

# function to perform OLS regression
# inputs: X matrix with column of 1s for intercept, Y matrix
def ols_reg(X,Y):
    # B_hat = inv(X'X) * (X'Y)
    B_hat = np.linalg.inv(X.T*X)*(X.T*Y)
    # residuals = Y - X*B_hat
    resids = Y - (X*B_hat)
    # estimate covariance matrix of error terms
    sig_u_hat = np.diag(np.squeeze(np.asarray(np.square(resids))))    
    # est SE(B_hat) = inv(X'X) * X' * sigma_u_hat * X * inv(X'X) (white SEs)
    sig_hat = (np.linalg.inv(X.T*X))*X.T*sig_u_hat*X*(np.linalg.inv(X.T*X))
    SE_B_hat = np.sqrt(sig_hat.diagonal())
    return [B_hat.T,SE_B_hat]
