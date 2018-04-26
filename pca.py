"""
Author: Rajkumar Conjeevaram Mohan
Date: 25.04.2018
Email: rajkumarcm@yahoo.com
Program: Principal Component Analysis
"""
import numpy as np
import preprocess as pre

def _covariance(data):
    """
    Computes the covariance of data in feature space
    and expects the input data to be centralised 
    i.e., zero mean
    :param data: [instances,features]
    :return: covariance of shape: [features,features]
    """

    return np.dot(data.T,data)

def _components(cov):
    D,V = np.linalg.eig(cov)
    values = D
    indices = np.argsort(values)[::-1]
    values = values[indices]
    V = V[:,indices]
    # V_mag = np.sqrt(np.sum(V**2,axis=0))
    # V /= V_mag
    values /= np.sum(values)
    return values,V

def _project_data(data,Basis,n_comp):
    """
    :param data: Data to be transformed. Shape: [instances,features] 
    :param Basis: array that describes the axes. Shape: [features,features]
    :param n_comp: Number of dimensions to retain
    :return: projected data of shape: [instances,n_comp]
    """
    return np.dot(data,Basis[:,:n_comp])

def project_data(data,new_dim):
    cov = _covariance(data)
    _,V = _components(cov)
    return _project_data(data,V,new_dim)

