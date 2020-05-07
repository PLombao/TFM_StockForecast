from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor

def get_ranked_correlation(df, target):
    df = pd.get_dummies(df, drop_first=True)
    rank = {'pearson':[], 
            'spearman':[], 
            'kendall':[]}
    
    for column in df.columns.drop(target):
        data = df[[column, target]].dropna()
            
        #Pearson
        cor, p = pearsonr(data[column], data[target])
        rank['pearson'].append([column, abs(cor), p, cor/abs(cor)])

        # Spearman
        cor, p = spearmanr(data[column], data[target])
        rank['spearman'].append([column, abs(cor), p, cor/abs(cor)])

        # Kendall Tau
        cor, p = kendalltau(data[column], data[target])
        rank['kendall'].append([column, abs(cor), p, cor/abs(cor)])
            
    # Transform data to df and order
    all_df = pd.DataFrame({"feature":df.columns.drop(target)})
    for key in list(rank):
        columns = ["feature"] + [x + "_" + key for x in ['correlation', 'pvalue', 'sign']]
        df = pd.DataFrame(data=rank[key],columns =columns)
        all_df = all_df.merge(df, on="feature")
            
    return all_df.sort_values("correlation_spearman")

def lasso_features(X, y, alpha=0.8):
    """Given a dataset, a list with the names of features and a target, optimizes a linear model
    using LASSO regularization and returns a list with the absolute value of the obtained
    coefficients. The returned list provides a ranked interpretation of the weights of each feature,
    which can be used for feature selection.
    Args:
        data (pd.DataFrame):    DataFrame with a dataset including features and a target label.
        features (list):        Names of the columns of the input DataFrame that contain features
                                for the linear model.
        target (str):           Name of the column of the input DataFrame that contain the target
                                for the linear model.
        alpha (float):          Constant multiplying the L1 term in the optimization objective.
    Returns:
        list:                   List of the absolute value of the obtained coefficients.
    """
    
    regr = linear_model.Lasso(alpha=alpha, max_iter=10000)
    regr.fit(X, y)
    rank_features = list(np.abs(regr.coef_))
    
    ranking = pd.DataFrame({"features":X.columns, 
                            "value":rank_features})
    return ranking.sort_values("value", ascending=False).reset_index(drop=True)

def get_rfe(X,y, estimator):
    selector = RFE(estimator, 1, step=1)
    selector = selector.fit(X, y)
    rank = selector.ranking_
    rank = pd.DataFrame({"features":X.columns, "rank":rank})
    return rank.sort_values("rank").reset_index(drop=True)

def get_sfm(X,y,estimator):
    selector = SelectFromModel(estimator).fit(X, y)
    
    coef = selector.estimator_.feature_importances_
    sfm = pd.DataFrame({"features":X.columns, "coef":coef})
    return sfm.sort_values("coef", ascending=False).reset_index(drop=True)
    
    

def compute_feature_selection(df, target, features, estimator):
    df = pd.get_dummies(df, drop_first=True)
    
    X = df[features]
    y = df[target]
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(data=X, columns=features)
    
    # Get correlation
    df = X.copy()
    df[target] = y
    
    corr = get_ranked_correlation(df, target)
    
    # Get Lasso
    lasso = lasso_features(X,y)
    
    # Get RFE
    rfe = get_rfe(X,y, estimator)
    
    # Get Model
    sfm = get_sfm(X,y,estimator)
    
    return corr, lasso, rfe, sfm