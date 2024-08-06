from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, validate_call
from typing import Callable, Optional, Union
from numpy.typing import NDArray
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

class DistanceOptions(str, Enum):
    """Implemented distances"""
    ovl = 'overlapCoef'
    bc = 'Bhattacharyya'
    h = 'Hellinger'
    l1 = 'L1-distance'

class Base(BaseModel):
    """Base Fuzzy C-means Model"""
    
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    numClust: int = Field(5, ge=1)
    maxIter: int = Field(150, ge=1, le=1000)
    mFuzzy: float = Field(2.0, ge=1.0)
    epsilon: float = Field(1e-5, ge=1e-9)
    h: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = False
    n_jobs: int = Field(1, ge=1)
    verbose: Optional[bool] = False
    distance: Optional[Union[DistanceOptions, Callable]] = DistanceOptions.ovl
    thetaIni: str = 'random'

    def _is_trained(self) -> bool:
        return self.trained

    def initializePrototype(self, Data: pd.DataFrame) -> pd.DataFrame:
        if self.thetaIni == 'random':
            idx = np.random.default_rng(self.random_state).permutation(Data.shape[1])[:self.numClust]
            return Data.iloc[:, idx]
        if self.thetaIni == 'partition':
            U0 = np.random.default_rng(self.random_state).random((self.numClust, Data.shape[1]))
            U0 /= U0.sum(axis=0)
            return pd.DataFrame((Data.values @ (U0 ** self.mFuzzy).T) / (U0 ** self.mFuzzy).sum(axis=1))
        if self.thetaIni == 'overlap':
            return self.initializeOverlap(Data)
        if self.thetaIni == 'kmeans++':
            return self.initializeKmeansPP(Data)
        raise ValueError("Invalid thetaIni parameter value. Must be 'random', 'partition', 'overlap', or 'kmeans++'.")

    def initializeOverlap(self, Data: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        theta = pd.DataFrame(np.zeros((Data.shape[0], self.numClust)))
        theta.iloc[:, 0] = Data.iloc[:, rng.integers(0, Data.shape[1])]
        for i in range(1, self.numClust):
            maxDist, attempts = -np.inf, 0
            while attempts < 1000:
                tt = rng.integers(0, Data.shape[1])
                farest = min(self._BhattacharyyaDist(theta.iloc[:, j], Data.iloc[:, tt], self.h) for j in range(i))
                if farest > maxDist:
                    maxDist, theta_tt = farest, tt
                    if maxDist > 0.9:
                        break
                attempts += 1
            if theta_tt != -1:
                theta.iloc[:, i] = Data.iloc[:, theta_tt]
            else:
                raise ValueError('Could not find a valid initial center. Try increasing max_attempts or adjusting min_distance_threshold.')
        return theta

    def initializeKmeansPP(self, Data: pd.DataFrame) -> pd.DataFrame:
        num_samples = Data.shape[1]
        centers = [Data.iloc[:, np.random.default_rng(self.random_state).integers(num_samples)]]
        for _ in range(1, self.numClust):
            distances = np.array([min(self._L1Dist(data, center, self.h) for center in centers) for data in Data.T.values])
            probabilities = distances / distances.sum()
            centers.append(Data.iloc[:, np.random.default_rng(self.random_state).choice(num_samples, p=probabilities)])
        return pd.DataFrame(centers).T

    def _distMat(self, Data: pd.DataFrame, fv: pd.DataFrame) -> NDArray:
        Wf = np.zeros((self.numClust, Data.shape[1]))
        for j in range(Data.shape[1]):
            for i in range(self.numClust):
                Wf[i, j] = self._Distance(fv.iloc[:, i], Data.iloc[:, j]) + np.finfo(float).eps
        return Wf ** 2

    def _Distance(self, f1: NDArray, f2: NDArray) -> float:
        if callable(self.distance):
            return self.distance(f1, f2, self.h)
        if self.distance == DistanceOptions.ovl:
            return self._overlapDist(f1, f2, self.h)
        if self.distance == DistanceOptions.h:
            return self._HellingerDist(f1, f2, self.h)
        if self.distance == DistanceOptions.bc:
            return self._BhattacharyyaDist(f1, f2, self.h)
        if self.distance == DistanceOptions.l1:
            return self._L1Dist(f1, f2, self.h)
        raise ValueError(f"Unknown distance method: {self.distance}")

    @staticmethod
    def _overlapDist(f1: NDArray, f2: NDArray, dx: float) -> float:
        return 1 - np.sum(np.minimum(f1, f2)) * dx

    @staticmethod
    def _HellingerDist(f1: NDArray, f2: NDArray, dx: float) -> float:
        return np.sqrt((np.sum(np.sqrt(f1) - np.sqrt(f2)) * dx) ** 2)
    
    @staticmethod
    def _BhattacharyyaDist(f1: NDArray, f2: NDArray, dx: float) -> float:
        return -np.log(np.sum(np.sqrt(f1 * f2)) * dx)

    @staticmethod
    def _L1Dist(f1: NDArray, f2: NDArray, dx: float) -> float:
        return np.sum(np.abs(f1 - f2) * dx)



class IFCM(Base):
    """Fuzzy C-means Model with additional features"""
    
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(self, Data: pd.DataFrame) -> None:
        """Train the fuzzy-c-means model"""
        self.numSample = Data.shape[1]
        self.rng = np.random.default_rng(self.random_state)

        fv = self.initializePrototype(Data)
        U = np.ones((self.numClust, self.numSample)) / self.numClust

        start_time = time.time()
        iter = 0

        while iter < self.maxIter:
            iter += 1

            self.Wf = self._distMat(Data, fv)
            self.fci = U.sum(axis=1)
            
            # Update partition matrix
            self.Uifcm = self._updatePartition(Data)
            # Calculate the cluster centers
            self.theta = self._updatePrototype(Data)

            ObjFun = np.sum(self.Uifcm * self.Wf / self.fci[:, np.newaxis])
            print(f'Iteration count = {iter}, obj. ifcm = {ObjFun:.6f}')

            # Check for convergence
            if np.linalg.norm(fv - self.theta, 1) < self.epsilon:
                break

            fv = self.theta.copy()
            U = self.Uifcm.copy()

        self.trained = True
        print(f"Training Time: {time.time() - start_time}")

    def _updatePartition(self, Data: pd.DataFrame) -> NDArray:
        """Soft predict of FCM"""
        Unew = np.zeros((self.numClust, self.numSample))
        for i in range(self.numClust):
            for j in range(self.numSample):
                numerator = self.fci[i] / (self.Wf[i, j] ** (2 / (self.mFuzzy - 1)))
                denominator = sum(self.fci[k] / (self.Wf[k, j] ** (2 / (self.mFuzzy - 1))) for k in range(self.numClust))
                Unew[i, j] = numerator / denominator
        return Unew

    def _updatePrototype(self, Data: pd.DataFrame) -> pd.DataFrame:
        """Update the cluster centers"""
        return pd.DataFrame((Data.values @ (self.Uifcm ** self.mFuzzy).T) / (self.Uifcm ** self.mFuzzy).sum(axis=1))
    
    def defuzzication(self, Data: pd.DataFrame) -> NDArray:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X (pd.DataFrame): New data to predict.

        Raises:
            ReferenceError: If it called without the model being trained.

        Returns:
            NDArray: Index of the cluster each sample belongs to.
        """
        if self._is_trained():
            return self._updatePartition(Data).argmax(axis=0)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )
    def predict(self, Data: pd.DataFrame) -> NDArray:
        """Predict membership values for new data"""
        if not self._is_trained():
            raise ReferenceError("You need to train the model. Run `.fit()` method to this.")
        distMat = self._distMat(Data, self.theta)
        return np.argmin(distMat, axis=0)
    
    @property
    def partitionCoeff(self) -> float:
        """Partition coefficient"""
        if self._is_trained():
            return np.mean(self.Uifcm**2)
        raise ReferenceError("You need to train the model. Run `.fit()` method to this.")

    @property
    def partitionEntropyCoef(self) -> float:
        if self._is_trained():
            return -np.mean(self.Uifcm * np.log2(self.Uifcm))
        raise ReferenceError("You need to train the model. Run `.fit()` method to this.")


class FCM(Base):
    """Fuzzy C-means Model with additional features"""
    
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(self, Data: pd.DataFrame) -> None:
        """Train the fuzzy-c-means model"""
        self.numSample = Data.shape[1]
        self.rng = np.random.default_rng(self.random_state)

        fv = self.initializePrototype(Data)
        U = np.ones((self.numClust, self.numSample)) / self.numClust

        start_time = time.time()
        iter = 0

        while iter < self.maxIter:
            iter += 1

            self.Wf = self._distMat(Data, fv)
            
            # Update partition matrix
            self.Ufcm = self._updatePartition(Data)
            # Calculate the cluster centers
            self.theta = self._updatePrototype(Data)

            ObjFun = np.sum(self.Ufcm * self.Wf)
            print(f'Iteration count = {iter}, obj. fcm = {ObjFun:.6f}')

            # Check for convergence
            if np.linalg.norm(fv - self.theta, 1) < self.epsilon:
                break

            fv = self.theta.copy()

        self.trained = True
        print(f"Training Time: {time.time() - start_time}")

    def _updatePartition(self, Data: pd.DataFrame) -> NDArray:
        """Soft predict of FCM"""
        Unew = np.zeros((self.numClust, self.numSample))
        for i in range(self.numClust):
            for j in range(self.numSample):
                numerator = 1 / (self.Wf[i, j] ** (2 / (self.mFuzzy - 1)))
                denominator = sum(1 / (self.Wf[k, j] ** (2 / (self.mFuzzy - 1))) for k in range(self.numClust))
                Unew[i, j] = numerator / denominator
        return Unew

    def _updatePrototype(self, Data: pd.DataFrame) -> pd.DataFrame:
        """Update the cluster centers"""
        return pd.DataFrame((Data.values @ (self.Ufcm ** self.mFuzzy).T) / (self.Ufcm ** self.mFuzzy).sum(axis=1))

    def defuzzication(self, Data: pd.DataFrame) -> NDArray:
        """Predict the closest cluster each sample in X belongs to"""
        if self._is_trained():
            return self._updatePartition(Data).argmax(axis=0)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )
    
    def predict(self, Data: pd.DataFrame) -> NDArray:
        """Predict membership values for new data"""
        if not self._is_trained():
            raise ReferenceError("You need to train the model. Run `.fit()` method to this.")
        distMat = self._distMat(Data, self.theta)
        return np.argmin(distMat, axis=0)
    
    @property
    def partitionCoeff(self) -> float:
        """Partition coefficient"""
        if self._is_trained():
            return np.mean(self.Ufcm**2)
        raise ReferenceError("You need to train the model. Run `.fit()` method to this.")

    @property
    def partitionEntropyCoef(self) -> float:
        if self._is_trained():
            return -np.mean(self.Ufcm * np.log2(self.Ufcm))
        raise ReferenceError("You need to train the model. Run `.fit()` method to this.")


class KMEAN(Base):
    """K-means Model with additional features"""

    def fit(self, Data: pd.DataFrame) -> None:
        """Train the K-means model"""
        self.numSample = Data.shape[1]
        self.rng = np.random.default_rng(self.random_state)

        # Initialize cluster centers
        self.theta = self.initializePrototype(Data)
        self.U = np.zeros((self.numClust, self.numSample), dtype=int)

        start_time = time.time()
        iter = 0

        while iter < self.maxIter:
            iter += 1

            self.Wf = self._distMat(Data, self.theta)
            labels = self._assignLabels(Data)

            new_theta = self._updatePrototype(Data, labels)

            if np.linalg.norm(self.theta - new_theta, 1) < self.epsilon:
                break

            self.theta = new_theta
            self.U = labels

        self.trained = True
        print(f"Training Time: {time.time() - start_time} with {iter} iterations")
    
    def _updatePrototype(self, Data: pd.DataFrame, labels) -> pd.DataFrame:
        """Update the cluster centers based on the current data and labels."""

        centroids = pd.DataFrame(np.zeros((Data.shape[0], self.numClust)))

        for k in range(self.numClust):
            points_in_cluster = Data.loc[:, labels == k]

            if points_in_cluster.shape[0] > 0:
                centroids[k] = points_in_cluster.mean(axis=1)

        return centroids    

    def _assignLabels(self, Data: pd.DataFrame) -> NDArray:
        return np.argmin(self.Wf, axis=0)


    def predict(self, Data: pd.DataFrame) -> NDArray:
        """Predict cluster assignments for new data"""
        if not self._is_trained():
            raise ReferenceError("You need to train the model. Run `.fit()` method to this.")
        distMat = self._distMat(Data, self.theta)
        return np.argmin(distMat, axis=0)
