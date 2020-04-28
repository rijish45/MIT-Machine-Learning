"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, var, pi = mixture  
    K = mu.shape[0]
    pre_exp = (2*np.pi*var)**(d/2)
    post = np.linalg.norm(X[:,None] - mu, ord=2, axis=2)**2   
    post = np.exp(-post/(2*var))
    post = post/pre_exp     
    numerator = post*pi
    denominator = np.sum(numerator, axis=1).reshape(-1,1) 
    post = numerator/denominator      
    log_lh = np.sum(np.log(denominator), axis=0).item()    
    return post, log_lh
   


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    nj = np.sum(post, axis=0)  
    pi = nj/n   
    mu = (post.T @ X)/nj.reshape(-1,1)  
    norms = np.linalg.norm(X[:, None] - mu, ord=2, axis=2)**2    
    var = np.sum(post*norms, axis=0)/(nj*d)    
    return GaussianMixture(mu, var, pi)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
   	old_log_lh = None
    new_log_lh = None 
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):
        old_log_lh = new_log_lh
        post, new_log_lh = estep(X, mixture)
        mixture = mstep(X, post)         
    return mixture, post, new_log_lh
