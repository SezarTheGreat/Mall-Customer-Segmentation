import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoftClusterer:
    """
    Performs Soft Clustering using GaussianMixture (GMM).
    Automatically selects the optimal K (number of clusters) using the Bayesian Information Criterion (BIC).
    Outputs a matrix of probability distributions across the K clusters.
    """

    def __init__(self, min_k: int = 2, max_k: int = 10, random_state: int = 42):
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state
        self.optimal_k = None
        self.model = None

    def fit_predict_proba(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Runs the BIC loop to find the optimal K, fits the best GMM model, 
        and returns the soft probabilities and the dominant cluster argmax.
        """
        logger.info(f"Starting BIC sweep for optimal K between {self.min_k} and {self.max_k}...")
        
        # Optimization: Sample representative rows if dataset is massive to prevent OOM/extreme runtimes.
        max_samples = 50000
        if X.shape[0] > max_samples:
            logger.info(f"Dataset shape {X.shape[0]} is massive. Sampling {max_samples} rows for GMM training and BIC sweep.")
            X_train = X.sample(n=max_samples, random_state=self.random_state)
        else:
            X_train = X

        lowest_bic = np.inf
        best_gmm = None
        
        for k in range(self.min_k, self.max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=self.random_state)
            gmm.fit(X_train)
            bic_score = gmm.bic(X_train)
            
            logger.info(f"K={k} | BIC={bic_score:.2f}")
            
            if bic_score < lowest_bic:
                lowest_bic = bic_score
                best_gmm = gmm
                self.optimal_k = k
        
        self.model = best_gmm
        logger.info(f"Optimal K selected: {self.optimal_k} with BIC: {lowest_bic:.2f}")
        
        # Predict soft probabilities for each cluster on the full dataset (fast)
        probabilities = self.model.predict_proba(X)
        
        prob_cols = [f"Cluster_{i}_Prob" for i in range(self.optimal_k)]
        prob_df = pd.DataFrame(probabilities, columns=prob_cols, index=X.index)
        
        # Also compute the dominant cluster assignment (argmax) for supervised layer later
        dominant_cluster = prob_df.idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))
        
        return prob_df, dominant_cluster

if __name__ == "__main__":
    # Test Clustering
    clusterer = SoftClusterer()
    test_X = pd.DataFrame(np.random.rand(100, 5))
    probs, labels = clusterer.fit_predict_proba(test_X)
    print(probs.head())
    print("Labels:", labels.head().values)
