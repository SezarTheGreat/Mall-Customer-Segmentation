import pandas as pd
import duckdb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestor:
    """
    DataIngestor handles the multi-source data ingestion pipeline.
    It reads and aggregates structural features from:
    1. Online Retail II (RFM)
    2. eCommerce Behavior (Interaction Metrics using DuckDB to avoid OOM)
    3. Instacart (Basket patterns)
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.ecommerce_path = self.base_dir / "eCommerce Behaviour data from multi category store"
        self.retail_path = self.base_dir / "Online Retail II Uci Datase"
        self.instacart_path = self.base_dir / "Instacar Market basket analysis"

    def process_ecommerce_data(self) -> pd.DataFrame:
        """
        Uses DuckDB to run out-of-core SQL aggregations on massive CSV files (~5GB+).
        Extracts user-level interaction metrics: views, carts, purchases, and sessions.
        """
        logger.info("Starting out-of-core ingestion for eCommerce data using DuckDB...")
        con = duckdb.connect(database=':memory:')
        
        file_glob = str(self.ecommerce_path / "*.csv")
        
        query = f"""
            SELECT 
                user_id::VARCHAR AS customer_id,
                COUNT(CASE WHEN event_type = 'view' THEN 1 END) AS total_views,
                COUNT(CASE WHEN event_type = 'cart' THEN 1 END) AS total_carts,
                COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS total_purchases,
                SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_ecommerce_spend,
                COUNT(DISTINCT user_session) AS total_sessions
            FROM read_csv_auto('{file_glob}', ignore_errors=true)
            WHERE user_id IS NOT NULL
            GROUP BY user_id
        """
        
        df = con.execute(query).df()
        
        # Calculate checkout abandonment rate safely
        df['checkout_abandonment_flag'] = (df['total_carts'] > df['total_purchases']).astype(int)
        
        logger.info(f"Successfully processed eCommerce data. Extracted features for {len(df)} users.")
        return df

    def process_retail_rfm(self) -> pd.DataFrame:
        """
        Loads the transactional logs to calculate Recency, Frequency, and Monetary Value (RFM) vectors.
        """
        logger.info("Ingesting Online Retail II data...")
        retail_file = self.retail_path / "online_retail_II.csv"
        
        df = pd.read_csv(retail_file)
        df.dropna(subset=['Customer ID'], inplace=True)
        
        # Standardize ID column name
        df.rename(columns={'Customer ID': 'customer_id'}, inplace=True)
        df['customer_id'] = df['customer_id'].astype(int).astype(str)
        
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalAmount'] = df['Quantity'] * df['Price']
        
        # Reference date for Recency (usually max date + 1 day)
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('customer_id').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,
            'Invoice': 'nunique',
            'TotalAmount': 'sum'
        }).reset_index()
        
        rfm.rename(columns={
            'InvoiceDate': 'Recency',
            'Invoice': 'Frequency',
            'TotalAmount': 'MonetaryValue'
        }, inplace=True)
        
        logger.info(f"Successfully computed RFM for {len(rfm)} users.")
        return rfm

    def process_instacart_data(self) -> pd.DataFrame:
        """
        Loads Instacart data to capture general order volume and patterns.
        """
        logger.info("Ingesting Instacart orders data...")
        orders_file = self.instacart_path / "orders.csv"
        
        df = pd.read_csv(orders_file)
        
        df.rename(columns={'user_id': 'customer_id'}, inplace=True)
        df['customer_id'] = df['customer_id'].astype(str)
        
        basket_features = df.groupby('customer_id').agg({
            'order_number': 'max',  # total orders
            'days_since_prior_order': 'mean' # average days between orders
        }).reset_index()
        
        basket_features.rename(columns={
            'order_number': 'instacart_total_orders',
            'days_since_prior_order': 'avg_days_between_orders'
        }, inplace=True)
        
        # Fill NaNs for users with only 1 order
        basket_features['avg_days_between_orders'].fillna(0, inplace=True)
        
        logger.info(f"Successfully processed Instacart data for {len(basket_features)} users.")
        return basket_features

    def get_unified_dataset(self) -> pd.DataFrame:
        """
        Merges all external engineered datasets into a unified primary training matrix.
        Performs OUTER joins using 'customer_id'.
        """
        logger.info("Starting unified data fusion...")
        
        rfm_df = self.process_retail_rfm()
        ecommerce_df = self.process_ecommerce_data()
        basket_df = self.process_instacart_data()
        
        # Merge datasets. Outer join to keep a comprehensive global user matrix
        merged_df = pd.merge(rfm_df, ecommerce_df, on='customer_id', how='outer')
        merged_df = pd.merge(merged_df, basket_df, on='customer_id', how='outer')
        
        # Fill missing numeric values with 0 for users who don't exist in a given dataset
        merged_df.fillna(0, inplace=True)
        
        logger.info(f"Final unified dataset created with shape: {merged_df.shape}")
        return merged_df

if __name__ == "__main__":
    # Test Ingestion 
    ingestor = DataIngestor(base_dir="..")
    df = ingestor.get_unified_dataset()
    print(df.head())
