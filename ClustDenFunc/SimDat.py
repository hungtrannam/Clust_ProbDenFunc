
import numpy as np
import pandas as pd


def simDatabnorm(mu_ranges, sig_values, grid, abnormal_params=None):
    
    """
    SimPDFAbnormal generates and plots PDFs for multiple sets of distributions
    with specified means and standard deviations, including optional
    "abnormal" distributions.

    Parameters:
    mu_ranges      : List of arrays of means for each set of distributions
    sig_values     : Array of standard deviations for all distributions
    grid           : Array of x-axis values for plotting the PDFs
    abnormal_params: List of tuples, each containing two lists:
                     - List 1: Array of means for the abnormal distribution
                     - List 2: Array of standard deviations for the abnormal distribution

    Returns:
    Data   : Matrix of generated PDFs without labels
    labels : Labels indicating the set to which each PDF belongs
    """
    from scipy.stats import norm

    num_groups = len(mu_ranges)
    denFuncs = []
    labels = []

    for group in range(num_groups):
        mu_range = mu_ranges[group]
        for mu in mu_range:
            f_single = norm.pdf(grid, mu, sig_values[group])
            denFuncs.append(f_single)
            labels.append(group)

    if abnormal_params is not None:
        for a_group, (mus, sigmas) in enumerate(abnormal_params):
            abnormal_pdf = np.zeros_like(grid)
            for mu, sigma in zip(mus, sigmas):
                abnormal_pdf += norm.pdf(grid, mu, sigma)
            denFuncs.append(abnormal_pdf)
            labels.append(num_groups)  

    Series = np.array(denFuncs).T

    DataFuncs = [pd.Series(Series[:, idx], name=f'pdf{idx+1}') for idx in range(Series.shape[1])]
    Data = pd.DataFrame(DataFuncs).T
    labels = np.array(labels)

    return Data, labels

def splitTrainTest(df, trueLabels, test_size=0.2, random_state=42, stratify=None):
    """
    split data into training and testing
    """
    from sklearn.model_selection import train_test_split
    df = df.T

    train_df, test_df, train_lbs, test_lbs = train_test_split(df, trueLabels,  
                                         test_size=test_size, 
                                         random_state=random_state, 
                                         stratify=stratify)

    return train_df.T, test_df.T, train_lbs.T, test_lbs.T