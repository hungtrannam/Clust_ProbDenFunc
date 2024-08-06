
def IFCM__(data, param):
    from const import np
    from ClustDenFunc.Toolbox import initializePrototype, integrationDenFunc

    if param['kClust'] <= 1:
        raise ValueError('The number of clusters (kClust) must be greater than 1 for clustering to be performed.')

    f = data
    iter = 0
    max_iter = param['maxIter']
    fm = param['mFuzzy']
    epsilon = param['epsilon']
    num_sample = f.shape[1]
    num_cluster = param['kClust']
    dx = param['h']

    fv = initializePrototype(f, param, num_cluster)
    
    U = np.ones((num_cluster, num_sample)) / num_cluster

    # Repeat FCM until convergence or max iterations
    while iter < max_iter:
        iter += 1

        # Calculate the distance between fv with fi PDFs
        Wf = np.zeros((num_cluster, num_sample))
        for j in range(num_sample):
            for i in range(num_cluster):
                Wf[i, j] = 1 - integrationDenFunc(np.minimum(fv[:, i], f[:, j]), dx) + 1e-10
        Wf **= 2

        fci = U.sum(axis=1)

        # Update partition matrix
        Unew = np.zeros((num_cluster, num_sample))
        for i in range(num_cluster):
            for j in range(num_sample):
                numerator = fci[i] / (Wf[i, j] ** (2 / (fm - 1)))
                denominator = sum(fci[k] / (Wf[k, j] ** (2 / (fm - 1))) for k in range(num_cluster))
                Unew[i, j] = numerator / denominator

        # Calculate the cluster centers
        fvnew = (f @ (Unew ** fm).T) / (Unew ** fm).sum(axis=1)

        ObjFun = np.sum(Unew * Wf / fci[:, np.newaxis])
        print(f'Iteration count = {iter}, obj. ifcm = {ObjFun:.6f}')

        # Check for convergence
        if np.linalg.norm(fv - fvnew, 1) < epsilon:
            break

        fv = fvnew
        U = Unew

    # Results
    # IDX = np.argmax(Unew, axis=0)
    # results = {
    #     'fuzzyPartition': {'U': Unew},
    #     'Data': {'fv': fvnew, 'Data': f},
    #     'iter': iter,
    #     'ObjFun': ObjFun,
    #     'Cluster': {'IDX': IDX},
    #     'Dist': {'D': Wf},
    #     'Imba': fci
    # }
    return Unew, fvnew, Wf