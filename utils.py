#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:10:29 2021

@author: rodrigo
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def hat_matrix(X, add_bias=True):
    """
    Compute hat matrix for design matrix X.
    
    :param np.array X: design matrix of dimensions (n x d), 
    where n is the number of observations and d is the number of
    features.
    :param bool add_bias: if matrix does not contain column of 1, use True
    """
    if add_bias:
        X = np.hstack([np.ones([len(X), 1]), X])
    
    A = np.matmul(X.T, X)
    
    LL = cho_factor(A)
    return np.matmul(X, cho_solve(LL, X.T))