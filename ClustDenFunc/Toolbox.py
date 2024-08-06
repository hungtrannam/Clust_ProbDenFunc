import numpy as np

def integrationDenFunc(Data, h, dim=1):
    
    # Compute the number of mesh elements by raising the mesh size 'h' to the power of 'dim' (dimensionality)
    mesh = h**dim
    
    # Calculate the solution by multiplying the number of mesh elements by the sum of function values 'fv'
    sol = mesh * np.sum(Data)
    
    return sol

def initializePrototype(Data, param, numCluster, fm=2.0, random_state = None):
    """
    Initialize the partition matrix with Fuzzy C-Means (FCM).

    Parameters:
    f (numpy.ndarray): The dataset.
    param (dict): Parameters dictionary containing 'FvIni' and 'h'.
    num_cluster (int): Number of clusters.
    num_sample (int): Number of samples.
    fm (float): Fuzziness parameter.

    Returns:
    numpy.ndarray: Initialized partition matrix.
    """
    numSample = Data.shape[1]

    if param['thetaIni'] == 'random':
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(numSample)[:numCluster]
        theta = Data[:, idx]
    elif param['thetaIni'] == 'partition':
        rng = np.random.RandomState(random_state)
        U0 = rng.rand(numCluster, numSample)
        U0 /= U0.sum(axis=0)
        theta = (Data @ (U0**fm).T) / (U0**fm).sum(axis=1)
    elif param['thetaIni'] == 'distance':
        rng = np.random.RandomState(random_state)
        theta = np.zeros((Data.shape[0], numCluster))
        tt = rng.randint(0, numSample)
        theta[:, 0] = Data[:, tt]
        for i in range(1, numCluster):
            maxDist = -np.inf
            theta_tt = -1
            attempts = 0
            while attempts < 1000:
                tt = np.random.randint(0, numSample)
                farest = np.inf
                for j in range(i):
                    distance = 1 - integrationDenFunc(np.sqrt(theta[:, j] * Data[:, tt]), param['h'])
                    farest = min(farest, distance)
                if farest > maxDist:
                    maxDist = farest
                    theta_tt = tt
                    if maxDist > 0.9:
                        break
                attempts += 1
            if theta_tt != -1:
                theta[:, i] = Data[:, theta_tt]
            else:
                raise ValueError('Could not find a valid initial center. Try increasing max_attempts or adjusting min_distance_threshold.')
    else:
        raise ValueError("Invalid FvIni parameter value. Must be type 'random', 'partition', or 'distance'.")
    
    return theta