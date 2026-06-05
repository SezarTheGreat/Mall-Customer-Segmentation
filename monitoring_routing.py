import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import mmh3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionMonitor:
    """
    Monitors data drift between inference batches and baseline training data 
    using the Kolmogorov-Smirnov (KS) test.
    """

    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold

    def check_drift(self, baseline_df: pd.DataFrame, inference_df: pd.DataFrame):
        """
        Contrasts the inference batch against the baseline distribution.
        Raises a warning if structural data drift is detected.
        """
        logger.info("Starting Data Drift check (KS Test)...")
        drift_detected = False
        
        # We check drift only on numeric columns that exist in both
        common_cols = set(baseline_df.select_dtypes(include='number').columns).intersection(
            set(inference_df.select_dtypes(include='number').columns)
        )
        
        drifted_features = []
        for col in common_cols:
            stat, p_value = ks_2samp(baseline_df[col].dropna(), inference_df[col].dropna())
            if p_value < self.p_value_threshold:
                drifted_features.append((col, p_value))
                drift_detected = True
                
        if drift_detected:
            logger.warning(f"STRUCTURAL DATA DRIFT DETECTED in {len(drifted_features)} features.")
            for feat, p_val in drifted_features[:5]:
                logger.warning(f" - {feat} (p-value: {p_val:.4f})")
            if len(drifted_features) > 5:
                logger.warning(f" - ... and {len(drifted_features) - 5} more.")
        else:
            logger.info("No structural data drift detected.")

class Router:
    """
    Deterministic user routing for A/B testing using MurmurHash3 (mmh3).
    Splits users into 90% Variant Group and 10% Control Baseline Group.
    """

    def __init__(self, salt: str = "customer_segmentation_v1"):
        self.salt = salt

    def route_user(self, user_id: str) -> str:
        """
        Returns 'Variant' (90%) or 'Control' (10%) deterministically based on user_id.
        """
        hash_val = mmh3.hash(str(user_id) + self.salt, signed=False)
        # Modulo 100 gives a uniformly distributed int between 0 and 99
        bucket = hash_val % 100
        
        if bucket < 90:
            return "Variant"
        else:
            return "Control"

    def apply_routing(self, df: pd.DataFrame, user_id_col: str = 'customer_id') -> pd.DataFrame:
        """
        Applies routing to a batch of users and appends the 'ab_group' column.
        """
        if user_id_col not in df.columns:
            logger.error(f"User ID column '{user_id_col}' not found for routing.")
            return df
            
        logger.info("Applying A/B routing to user base...")
        df['ab_group'] = df[user_id_col].apply(self.route_user)
        
        variant_count = (df['ab_group'] == 'Variant').sum()
        control_count = (df['ab_group'] == 'Control').sum()
        logger.info(f"Routing split: Variant ({variant_count}), Control ({control_count})")
        
        return df

if __name__ == "__main__":
    monitor = ProductionMonitor()
    base = pd.DataFrame({'f1': np.random.normal(0, 1, 1000)})
    inf = pd.DataFrame({'f1': np.random.normal(0.5, 1, 100)})
    monitor.check_drift(base, inf)
    
    router = Router()
    users = pd.DataFrame({'customer_id': [str(i) for i in range(1000)]})
    users = router.apply_routing(users)
    print(users['ab_group'].value_counts(normalize=True))
