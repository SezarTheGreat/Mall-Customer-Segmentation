import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentClassifier:
    """
    Supervised layer treating the GMM dominant cluster (argmax) as the categorical target.
    Trains a Random Forest classifier and uses its native feature importance for explainability.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=self.random_state
        )

    def train_and_explain(self, X: pd.DataFrame, y: pd.Series, top_k: int = 5) -> list[str]:
        """
        Trains the Random Forest classifier on the raw/enriched features X against cluster target y.
        Calculates feature importances and returns the top_k global features.
        """
        logger.info(f"Training Supervised Random Forest Classifier on {X.shape[0]} samples with {X.shape[1]} features.")
        
        X_numeric = X.select_dtypes(include=['number']).copy()
        
        leak_cols = [c for c in X_numeric.columns if 'Prob' in c or 'Cluster' in c or 'id' in c.lower()]
        X_numeric = X_numeric.drop(columns=leak_cols, errors='ignore').fillna(0)
        
        self.model.fit(X_numeric, y)
        logger.info("Random Forest training completed.")
        
        logger.info("Extracting feature importances for explainability...")
        
        importance_df = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': self.model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        top_features = importance_df.head(top_k)['feature'].tolist()
        
        logger.info(f"Top {top_k} Features driving segment assignment globally: {top_features}")
        return top_features

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X.select_dtypes(include=['number']).copy()
        leak_cols = [c for c in X_numeric.columns if 'Prob' in c or 'Cluster' in c or 'id' in c.lower()]
        X_numeric = X_numeric.drop(columns=leak_cols, errors='ignore').fillna(0)
        return self.model.predict(X_numeric)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X.select_dtypes(include=['number']).copy()
        leak_cols = [c for c in X_numeric.columns if 'Prob' in c or 'Cluster' in c or 'id' in c.lower()]
        X_numeric = X_numeric.drop(columns=leak_cols, errors='ignore').fillna(0)
        return self.model.predict_proba(X_numeric)

    def explain_local(self, X_sample: pd.DataFrame, predicted_class: int) -> pd.DataFrame:
        """
        Calculates local feature importance for a single customer sample using global importance as proxy.
        """
        try:
            X_numeric = X_sample.select_dtypes(include=['number']).copy()
            leak_cols = [c for c in X_numeric.columns if 'Prob' in c or 'Cluster' in c or 'id' in c.lower()]
            X_numeric = X_numeric.drop(columns=leak_cols, errors='ignore').fillna(0)

            global_importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': X_numeric.columns,
                'importance_value': global_importance
            })
            
            importance_df['abs_importance'] = importance_df['importance_value'].abs()
            importance_df = importance_df.sort_values(by='abs_importance', ascending=False).drop(columns=['abs_importance'])
            return importance_df
            
        except Exception as e:
            logger.error(f"Failed to generate random forest explanations: {e}")
            return pd.DataFrame(columns=['feature', 'importance_value'])

    def save_model(self, filepath: str):
        joblib.dump(self.model, filepath)
        
    @classmethod
    def load_model(cls, filepath: str):
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance