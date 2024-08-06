# ClustDenFunc

This repository contains Improved of Fuzzy C-means (FCM), Improved Fuzzy C-means (IFCM), and K-means clustering models. These models are built using Python and leverage the `pydantic` library for data validation and configuration management.

## Installation

To install the necessary packages, you can use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

``` python
import pandas as pd
from fcm import FCM

# Load your data into a DataFrame
mus = [
    np.linspace(-9, -5, num=5),  
    np.linspace(-1, 0, num=5*90), 
    np.linspace(5, 7, num = 5)
]
sigs = [1, 1, 1]
grid = np.linspace(-15, 12, 10000)
Data, trueLabels = simDatabnorm(mus, sigs, grid)


# Initialize the FCM model
fcm = FCM(numClust=3, maxIter=100, mFuzzy=2.0, epsilon=1e-9, h=0.1, distance='Bhattacharyya', thetaIni='kmeans++', verbose=True)

# Train the model
fcm.fit(data)

# Predict cluster membership for new data
predictions = fcm.predict(new_data)
```

## Model Parameters

  *  numClust: Number of clusters (default: 5)
  *  maxIter: Maximum number of iterations (default: 150)
*  mFuzzy: Fuzziness parameter for FCM and IFCM (default: 2.0)
 *   epsilon: Convergence threshold (default: 1e-5)
  *  h: Parameter for distance calculation (default: 1e-5)
   * random_state: Seed for random number generation (default: None)
  *  verbose: Verbose output during training (default: False)
  *  distance: Distance metric (overlapCoef, Bhattacharyya, Hellinger, L1-distance, or a custom function)
  *  thetaIni: Initialization method (random, partition, overlap, or kmeans++)

## Method
  *  fit(Data: pd.DataFrame): Train the model with the provided data.
  *  predict(Data: pd.DataFrame) -> NDArray: Predict cluster membership for new data.
  *  defuzzication(Data: pd.DataFrame) -> NDArray: Predict the closest cluster each sample belongs to.
  *  partitionCoeff: Property to get the partition coefficient.
  *  partitionEntropyCoef: Property to get the partition entropy coefficient.
### Distance Metric
  *  overlapCoef: Overlap coefficient distance.
   * Bhattacharyya: Bhattacharyya distance.
  *  Hellinger: Hellinger distance.
 *   L1-distance: L1 distance (Manhattan distance).


## References

  *  Fuzzy C-means (FCM): J.C. Bezdek, Pattern Recognition with Fuzzy Objective Function Algorithms, Springer, 1981.
  *  K-means: J. MacQueen, "Some Methods for classification and Analysis of Multivariate Observations," in Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability, 1967.




