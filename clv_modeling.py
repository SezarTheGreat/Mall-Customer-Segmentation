import pandas as pd
import logging
from lifetimes import BetaGeoFitter, GammaGammaFitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLVEstimator:
    """
    Probabilistic Customer Lifetime Value (CLV) Estimation using BG/NBD and Gamma-Gamma models.
    """

    def __init__(self, penalizer_coef: float = 0.01, months_to_predict: int = 12):
        self.penalizer_coef = penalizer_coef
        self.months_to_predict = months_to_predict
        self.bgf = BetaGeoFitter(penalizer_coef=self.penalizer_coef)
        self.ggf = GammaGammaFitter(penalizer_coef=self.penalizer_coef)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the BG/NBD and Gamma-Gamma submodels and estimates 12-month CLV.
        Expects df to contain 'Frequency', 'Recency', 'MonetaryValue', and 'T' (Customer Age in days).
        """
        logger.info("Starting CLV Estimation using BG/NBD and Gamma-Gamma models...")
        
        # Ensure required columns exist, infer T if missing
        required_cols = ['Frequency', 'Recency', 'MonetaryValue']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required RFM column: {col}")
                raise ValueError(f"Missing required RFM column: {col}")
                
        if 'T' not in df.columns:
            logger.warning("'T' (Customer Age) column missing. Inferring T as Recency + 30 days for modeling purposes.")
            df['T'] = df['Recency'] + 30
            
        # Filter out users with 0 frequency as BG/NBD and GammaGamma require repeat buyers
        repeat_buyers = df[df['Frequency'] > 0].copy()
        one_time_buyers = df[df['Frequency'] == 0].copy()
        
        if repeat_buyers.empty:
            logger.warning("No repeat buyers found to fit the model. Setting CLV to 0.")
            df['expected_12m_clv'] = 0.0
            df['expected_active_probability'] = 0.0
            return df
            
        # 1. Fit BG/NBD
        logger.info("Fitting BG/NBD model...")
        self.bgf.fit(repeat_buyers['Frequency'], repeat_buyers['Recency'], repeat_buyers['T'])
        
        repeat_buyers['expected_active_probability'] = self.bgf.conditional_probability_alive(
            repeat_buyers['Frequency'], repeat_buyers['Recency'], repeat_buyers['T']
        )
        
        # 2. Fit Gamma-Gamma (Note: monetary value must be strictly positive)
        logger.info("Fitting Gamma-Gamma model...")
        valid_monetary = repeat_buyers[repeat_buyers['MonetaryValue'] > 0]
        
        if valid_monetary.empty:
             repeat_buyers['expected_12m_clv'] = 0.0
        else:
             self.ggf.fit(valid_monetary['Frequency'], valid_monetary['MonetaryValue'])
             
             # Calculate 12-month (approx 365 days) CLV
             # Discount rate usually applied, here we use default monthly 0.01
             repeat_buyers['expected_12m_clv'] = self.ggf.customer_lifetime_value(
                 self.bgf,
                 repeat_buyers['Frequency'],
                 repeat_buyers['Recency'],
                 repeat_buyers['T'],
                 repeat_buyers['MonetaryValue'],
                 time=self.months_to_predict, # months
                 discount_rate=0.01 
             )
        
        # Merge back with one-time buyers (assign 0 CLV to them)
        one_time_buyers['expected_active_probability'] = 0.0
        one_time_buyers['expected_12m_clv'] = 0.0
        
        result_df = pd.concat([repeat_buyers, one_time_buyers])
        
        # Re-align with original dataframe index
        df['expected_12m_clv'] = result_df['expected_12m_clv']
        df['expected_active_probability'] = result_df['expected_active_probability']
        
        # Fill any remaining NaNs with 0
        df['expected_12m_clv'] = df['expected_12m_clv'].fillna(0)
        df['expected_active_probability'] = df['expected_active_probability'].fillna(0)
        
        logger.info("CLV Estimation completed.")
        return df

if __name__ == "__main__":
    estimator = CLVEstimator()
    test_df = pd.DataFrame({
        'Frequency': [10, 0, 5],
        'Recency': [100, 10, 50],
        'MonetaryValue': [150.0, 20.0, 300.0],
        'T': [120, 15, 60]
    })
    res = estimator.fit_predict(test_df)
    print(res)
