# Define common functions to be used by XGBoost and Extra Trees models

from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Obtain sorted feature list based on importance
def get_feature_ranking(feature_scores):
    indices = np.nonzero(feature_scores)[0].tolist()
    scores = np.abs(feature_scores[indices]).tolist() 
    sorted_scores, sorted_features = zip(*sorted(zip(scores, indices),reverse=True))
    return list(sorted_features)

# Root Mean Squared Error
def RMSE(y_true, y_hat):
    error = np.sqrt(mean_squared_error(y_true, y_hat))
    return error

# Train model using 10 fold CV
def train_model(X,Y,model):
    scores = cross_validate(model, X, Y, cv=10, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(np.abs(scores['test_score']))
    rmse = rmse_scores.mean()
    return rmse

# Backward recursive feature selection
def wrapper_feature_selector(X,Y,model,subset=np.arange(0,22).tolist()):
    sel = subset.copy() # # Selected features
    overall_error = train_model(X[:,sel],Y,model)
    while len(sel) != 0:
        # Select candidate
        cand_error = 1e10 # Assign a big number
        for cand in sel:
            features = sel.copy()
            features.remove(cand)
            if len(features) > 1:
                new_error = train_model(X[:,features],Y,model)
            else:
                new_error = train_model(X[:,features[0]].reshape(-1,1),Y,model)
            if new_error < cand_error:
                selected_candidate = cand
                cand_error = new_error
        if overall_error < cand_error:
            # Stop if the new candidate doesn’t
            # improve the assessment of the
            # previously selected candidates
            break
        else:
            overall_error = cand_error
            sel.remove(selected_candidate)
    rmse = train_model(X[:,sel],Y,model)
    # Return selected features and best rmse
    return [sel, rmse]


