import logging
import pandas as pd
import warnings
from pathlib import Path

# Suppress specific warnings for cleaner logs
warnings.filterwarnings('ignore')

from ingestion import DataIngestor
from enrichment import DataEnricher
from dimensionality_reduction import DimensionalityReducer
from clustering import SoftClusterer
from clv_modeling import CLVEstimator
from supervised_explainability import SegmentClassifier
from monitoring_routing import ProductionMonitor, Router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Main orchestrator that stitches the entire customer segmentation pipeline together.
    Phases:
    1. Data Ingestion (DuckDB/Pandas)
    2. Data Enrichment (FRED, Open-Meteo)
    3. Probabilistic CLV Modeling (BG/NBD & Gamma-Gamma)
    4. Dimensionality Reduction (StandardScaler + PCA)
    5. Soft Clustering (GMM + BIC)
    6. Supervised Explainability (Random Forest)
    7. Routing and Monitoring
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.ingestor = DataIngestor(base_dir=self.base_dir)
        self.enricher = DataEnricher()
        self.clv_estimator = CLVEstimator(months_to_predict=12)
        self.reducer = DimensionalityReducer(variance_threshold=0.95)
        self.clusterer = SoftClusterer(min_k=2, max_k=10)
        self.classifier = SegmentClassifier()
        self.router = Router()
        self.monitor = ProductionMonitor(p_value_threshold=0.05)
        
    def run_training_pipeline(self):
        logger.info("=== STARTING PIPELINE EXECUTION ===")
        
        # Phase 1: Ingestion
        df = self.ingestor.get_unified_dataset()
        if df.empty:
            logger.error("Unified dataset is empty. Aborting pipeline.")
            return None
            
        # Phase 1.b: Enrichment
        df = self.enricher.enrich_data(df)
        
        # Phase 3: Probabilistic CLV Estimation
        # (Executed before PCA so CLV can be a direct feature if desired)
        df = self.clv_estimator.fit_predict(df)
        
        # Store original customer IDs for later routing and joining
        customer_ids = df['customer_id']
        
        # Phase 1.c: Dimensionality Reduction
        reduced_df = self.reducer.fit_transform(df)
        
        # Phase 2: Soft Clustering Model Layer
        prob_df, target_labels = self.clusterer.fit_predict_proba(reduced_df)
        
        # Map back to IDs
        df['Segment_Cluster_ID'] = target_labels
        
        # Phase 4: Supervised Ensemble & Explainability
        # We train XGBoost on the Reduced PCA features OR the raw enriched features. 
        # For Explainability, raw numerical features are better so SHAP outputs readable names.
        raw_numerical = df.select_dtypes(include='number').drop(columns=['Segment_Cluster_ID'], errors='ignore')
        top_features = self.classifier.train_and_explain(raw_numerical, target_labels, top_k=5)
        
        # Phase 4.b: Serialize all submodels and estimators
        import joblib
        import json
        
        # Save CLV models (dumping params_ Series to bypass lambda pickling limitations in lifetimes)
        joblib.dump(self.clv_estimator.bgf.params_, Path(self.base_dir) / 'clv_bgf_params.pkl')
        joblib.dump(self.clv_estimator.ggf.params_, Path(self.base_dir) / 'clv_ggf_params.pkl')
        logger.info("Saved CLV model parameters ('clv_bgf_params.pkl', 'clv_ggf_params.pkl').")
        
        # Save Dimensionality Reducer
        joblib.dump(self.reducer.pipeline, Path(self.base_dir) / 'pca_pipeline.pkl')
        logger.info("Saved Dimensionality Reducer (PCA) pipeline ('pca_pipeline.pkl').")
        
        # Save GMM model
        joblib.dump(self.clusterer.model, Path(self.base_dir) / 'gmm_model.pkl')
        logger.info("Saved GMM soft clustering model ('gmm_model.pkl').")
        
        # Save XGBoost Classifier
        self.classifier.save_model(str(Path(self.base_dir) / 'xgb_classifier.pkl'))
        
        # Save the list of raw numerical feature names
        feature_names = raw_numerical.columns.tolist()
        with open(Path(self.base_dir) / 'feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        logger.info(f"Saved feature names list ({len(feature_names)} features) to 'feature_names.json'.")
        
        # Phase 5: Routing
        df['customer_id'] = customer_ids
        df = self.router.apply_routing(df)
        
        logger.info("=== PIPELINE EXECUTION COMPLETED ===")
        logger.info(f"Final Data Shape: {df.shape}")
        
        # Save a representative baseline sample for web app exploration and future drift monitoring
        output_path = Path(self.base_dir) / "baseline_training_data.parquet"
        if df.shape[0] > 100000:
            logger.info("Sampling 100,000 rows for the baseline parquet to keep the web app responsive and fast.")
            baseline_sample = df.sample(100000, random_state=42)
            baseline_sample.to_parquet(output_path, index=False)
        else:
            df.to_parquet(output_path, index=False)
        logger.info(f"Saved baseline data to {output_path}")
        
        return df

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator(base_dir=".")
    final_df = orchestrator.run_training_pipeline()
