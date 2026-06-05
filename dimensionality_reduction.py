import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DimensionalityReducer:
    """
    Constructs a dimensionality reduction pipeline applying StandardScaler 
    followed by PCA to automatically retain 95% of cumulative explained variance.
    """

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.variance_threshold, random_state=42))
        ])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler and PCA on the given DataFrame and returns the reduced components.
        Filters out non-numeric columns and identifier columns automatically.
        """
        logger.info(f"Starting Dimensionality Reduction (Target Variance: {self.variance_threshold*100}%)")
        
        # Select only numeric features, dropping identifiers if they exist
        features = df.select_dtypes(include=['number'])
        
        # Explicitly drop common ID columns if they slipped through
        id_cols = [c for c in features.columns if 'id' in c.lower()]
        features = features.drop(columns=id_cols, errors='ignore')
        
        # Handle any residual NaNs safely before PCA
        features = features.fillna(0)
        
        reduced_array = self.pipeline.fit_transform(features)
        
        n_components = self.pipeline.named_steps['pca'].n_components_
        logger.info(f"PCA completed. Retained {n_components} components to explain {self.variance_threshold*100}% variance.")
        
        # Convert back to DataFrame
        component_cols = [f"PC{i+1}" for i in range(reduced_array.shape[1])]
        reduced_df = pd.DataFrame(reduced_array, columns=component_cols, index=df.index)
        
        # We also want to return the raw df with the PCs concatenated, 
        # but typical clustering pipelines just take the reduced vectors.
        # We will return both or just the reduced df. For clean architecture, we return the reduced features.
        return reduced_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data based on the previously fitted pipeline.
        """
        features = df.select_dtypes(include=['number'])
        id_cols = [c for c in features.columns if 'id' in c.lower()]
        features = features.drop(columns=id_cols, errors='ignore').fillna(0)
        
        reduced_array = self.pipeline.transform(features)
        component_cols = [f"PC{i+1}" for i in range(reduced_array.shape[1])]
        
        return pd.DataFrame(reduced_array, columns=component_cols, index=df.index)

if __name__ == "__main__":
    # Test PCA
    import numpy as np
    reducer = DimensionalityReducer()
    test_data = pd.DataFrame(np.random.rand(100, 50))
    res = reducer.fit_transform(test_data)
    print(res.head())
