import gradio as gr
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifetimes import BetaGeoFitter, GammaGammaFitter
from supervised_explainability import SegmentClassifier

# Set dark theme for matplotlib globally
plt.rcParams.update({
    'grid.color': '#1f2937',
    'text.color': '#f3f4f6',
    'axes.labelcolor': '#9ca3af',
    'axes.edgecolor': '#1f2937',
    'xtick.color': '#9ca3af',
    'ytick.color': '#9ca3af',
    'figure.facecolor': '#070a13',
    'axes.facecolor': '#0d111e',
})

# Load the models, dataset and metadata
try:
    df_baseline = pd.read_parquet('baseline_training_data.parquet')
    feature_names = json.load(open('feature_names.json'))
    
    # Load CLV models
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.params_ = joblib.load('clv_bgf_params.pkl')
    bgf.predict = bgf.conditional_expected_number_of_purchases_up_to_time
    
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.params_ = joblib.load('clv_ggf_params.pkl')
    
    # Load PCA and GMM
    pca_pipeline = joblib.load('pca_pipeline.pkl')
    gmm_model = joblib.load('gmm_model.pkl')
    
    # Load XGBoost Segment Classifier
    classifier = SegmentClassifier.load_model('xgb_classifier.pkl')
    
    # Calculate dataset averages for default mock enrichment values
    defaults = {
        'CPIAUCSL': df_baseline['CPIAUCSL'].mean() if 'CPIAUCSL' in df_baseline.columns else 218.0,
        'RSXFS': df_baseline['RSXFS'].mean() if 'RSXFS' in df_baseline.columns else 350000.0,
        'PSAVERT': df_baseline['PSAVERT'].mean() if 'PSAVERT' in df_baseline.columns else 6.0,
    }
except Exception as e:
    print(f"Error loading models or dataset: {e}")
    df_baseline = None
    bgf = ggf = pca_pipeline = gmm_model = classifier = None
    defaults = {}

# Custom cluster names and details
CLUSTER_INFO = {
    0: {"name": "VIP / High-Value Loyalists", "desc": "Premium shoppers with high CLV, high repeat transactions, and balanced session activity.", "color": "#a78bfa"},
    1: {"name": "Inactive / Churned Customers", "desc": "Accounts with very high recency and near-zero active probability. Re-engagement needed.", "color": "#f87171"},
    2: {"name": "Instacart Bulk Grocery Buyers", "desc": "Routine buyers with high order counts and low days between orders. Consistent routine shoppers.", "color": "#34d399"},
    3: {"name": "Window Shoppers (High Carts, Low Purchase)", "desc": "High session views and carts, but low purchase counts and high cart abandonment rates.", "color": "#fbbf24"},
    4: {"name": "Weather-Sensitive Impulsive Buyers", "desc": "High weather-precipitation sensitivity flag. Driven by physical rain/precipitation states.", "color": "#f472b6"},
    5: {"name": "Casual One-Time Retail Buyers", "desc": "Low purchase frequency, high order values on single visits, and stable retail shopping profile.", "color": "#818cf8"},
    6: {"name": "Sensible / Standard Spenders", "desc": "Moderate transactional values, highly deliberate shopping paths, and balanced browsing habits.", "color": "#60a5fa"},
    7: {"name": "High-Intent Cart Abandoners", "desc": "High cart and browse history but high abandoned checkout rate. Re-target with promo vouchers.", "color": "#fb7185"},
    8: {"name": "New / Low-Engagement Registrations", "desc": "Recently created profiles with low transaction history, low sessions, and neutral activity.", "color": "#9ca3af"}
}

def make_local_shap_plot(importance_df, segment_name):
    """
    Plots a custom horizontal bar chart of local SHAP feature importances.
    """
    if importance_df.empty:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort for plotting (largest positive/negative values at the top)
    plot_df = importance_df.head(10).iloc[::-1]
    
    colors = ['#f43f5e' if val < 0 else '#8b5cf6' for val in plot_df['importance_value']]
    
    bars = ax.barh(plot_df['feature'], plot_df['importance_value'], color=colors, edgecolor='none', height=0.6)
    
    # Customize grid and spines
    ax.axvline(x=0, color='#ffffff', alpha=0.2, linestyle='-', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Title and labels
    ax.set_title(f"SHAP Values driving prediction of:\n{segment_name}", fontweight='bold', pad=15, fontsize=12)
    ax.set_xlabel("SHAP Influence (Direction & Strength)", labelpad=10)
    
    plt.tight_layout()
    return fig

def make_pca_scatter_plot(features_df, predicted_cluster, cluster_color):
    """
    Reduces the target features to PCA space and overlays it on the baseline population.
    """
    if df_baseline is None or pca_pipeline is None:
        return None
        
    try:
        # Get baseline PCs
        numeric_cols = df_baseline.select_dtypes(include='number').columns
        id_cols = [c for c in numeric_cols if 'id' in c.lower()]
        X_base = df_baseline[numeric_cols].drop(columns=id_cols + ['Segment_Cluster_ID'], errors='ignore').fillna(0)
        
        # Ensure identical column order
        base_features = X_base.columns.tolist()
        features_ordered = features_df[base_features].fillna(0)
        
        # Apply fitted scaler + PCA
        base_pca = pca_pipeline.transform(X_base)
        sample_pca = pca_pipeline.transform(features_ordered)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Draw background baseline points in subtle dark gray
        ax.scatter(
            base_pca[:, 0], 
            base_pca[:, 1], 
            color='#ffffff', 
            s=12, 
            alpha=0.06,
            label='Baseline Customers',
            edgecolor='none'
        )
        
        # Plot the target customer as a glowing star
        ax.scatter(
            sample_pca[0, 0],
            sample_pca[0, 1],
            color=cluster_color,
            marker='*',
            s=450,
            edgecolors='#ffffff',
            linewidths=1.8,
            label='Current Customer',
            zorder=10
        )
        
        # Add glow effect around the star
        ax.scatter(
            sample_pca[0, 0],
            sample_pca[0, 1],
            color=cluster_color,
            marker='o',
            s=1000,
            alpha=0.25,
            edgecolors='none',
            zorder=9
        )
        
        ax.set_title("Customer Position in 2D PCA Latent Space", fontweight='bold', pad=15)
        ax.set_xlabel("Principal Component 1 (PC1)")
        ax.set_ylabel("Principal Component 2 (PC2)")
        ax.grid(True, linestyle='--', alpha=0.15)
        ax.legend(loc='upper right', framealpha=0.6, facecolor='#0d111e', edgecolor='#ffffff')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error drawing PCA plot: {e}")
        return None

def process_lookup(customer_id):
    """
    Looks up a real customer in baseline parquet and returns metrics + predictions + charts.
    """
    if df_baseline is None or classifier is None:
        return "System error: Models or baseline parquet not loaded.", "", "", "", "", None, None
        
    try:
        row = df_baseline[df_baseline['customer_id'] == str(customer_id)]
        if row.empty:
            return f"Customer ID '{customer_id}' not found in the baseline dataset.", "", "", "", "", None, None
            
        # Extract features for XGBoost prediction
        # XGBoost expects the raw numeric features in exactly the same columns
        X_sample = row[feature_names].copy()
        
        # Perform Segment Prediction
        pred_cluster = int(classifier.predict(X_sample)[0])
        cluster_data = CLUSTER_INFO.get(pred_cluster, {"name": f"Segment {pred_cluster}", "desc": "Unknown segment characteristics.", "color": "#9ca3af"})
        
        # Extract specific metrics
        rfm_str = f"📅 Recency: {int(row['Recency'].iloc[0])} days | 🔄 Frequency: {int(row['Frequency'].iloc[0])} purchases | 💰 Monetary: ${row['MonetaryValue'].iloc[0]:,.2f}"
        
        ecom_str = f"👁️ Views: {int(row['total_views'].iloc[0])} | 🛒 Carts: {int(row['total_carts'].iloc[0])} | 🛍️ Purchases: {int(row['total_purchases'].iloc[0])} | 🚪 Sessions: {int(row['total_sessions'].iloc[0])}"
        
        clv_str = f"🎯 Expected 12m CLV: ${row['expected_12m_clv'].iloc[0]:,.2f} | 📈 Active Prob: {row['expected_active_probability'].iloc[0]*100:.1f}% | 🔀 Group: {row['ab_group'].iloc[0]}"
        
        desc_html = f"""
        <div style="padding: 15px; border-left: 5px solid {cluster_data['color']}; background: rgba(255,255,255,0.02); border-radius: 4px;">
            <h3 style="margin: 0; color: {cluster_data['color']}; font-size: 1.3em;">{cluster_data['name']}</h3>
            <p style="margin: 8px 0 0 0; color: #d1d5db; line-height: 1.5;">{cluster_data['desc']}</p>
        </div>
        """
        
        # Generate explanations & PCA scatter
        importance_df = classifier.explain_local(X_sample, pred_cluster)
        fig_shap = make_local_shap_plot(importance_df, cluster_data['name'])
        fig_pca = make_pca_scatter_plot(X_sample, pred_cluster, cluster_data['color'])
        
        return rfm_str, ecom_str, clv_str, desc_html, fig_shap, fig_pca
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error running lookup inference: {e}", "", "", "", None, None

def process_simulation(recency, frequency, monetary, views, carts, purchases, sessions, abandon, instacart_orders, days_between_orders, precip):
    """
    Computes lifetimes CLV predictions, constructs features, fits XGBoost segment classifier, and generates SHAP + PCA plots.
    """
    if classifier is None or bgf is None or ggf is None:
        return "System error: Predictors are not loaded.", "", "", "", None, None
        
    try:
        # Construct raw inputs
        precip_flag = 1 if precip == "Yes" else 0
        abandon_flag = 1 if abandon == "Yes" else 0
        T = recency + 30
        
        # Calculate lifetimes CLV predictions
        if frequency == 0:
            active_prob = 0.0
            expected_clv = 0.0
        else:
            freq_s = pd.Series([frequency])
            rec_s = pd.Series([recency])
            T_s = pd.Series([T])
            mon_s = pd.Series([monetary])
            
            active_prob_raw = bgf.conditional_probability_alive(freq_s, rec_s, T_s)
            if hasattr(active_prob_raw, "iloc"):
                active_prob = float(active_prob_raw.iloc[0])
            elif hasattr(active_prob_raw, "item"):
                active_prob = float(active_prob_raw.item())
            else:
                active_prob = float(active_prob_raw[0])
            
            clv_raw = ggf.customer_lifetime_value(bgf, freq_s, rec_s, T_s, mon_s, time=12, discount_rate=0.01)
            if hasattr(clv_raw, "iloc"):
                expected_clv = float(clv_raw.iloc[0])
            elif hasattr(clv_raw, "item"):
                expected_clv = float(clv_raw.item())
            else:
                expected_clv = float(clv_raw[0])
            
        # Build features dict using defaults for macro factors
        input_data = {
            'Recency': recency,
            'Frequency': frequency,
            'MonetaryValue': monetary,
            'total_views': views,
            'total_carts': carts,
            'total_purchases': purchases,
            'total_ecommerce_spend': purchases * monetary,
            'total_sessions': sessions,
            'checkout_abandonment_flag': abandon_flag,
            'instacart_total_orders': instacart_orders,
            'avg_days_between_orders': days_between_orders,
            'purchased_during_precipitation': precip_flag,
            'T': T,
            'expected_active_probability': active_prob,
            'expected_12m_clv': expected_clv,
            'CPIAUCSL': defaults.get('CPIAUCSL', 218.0),
            'RSXFS': defaults.get('RSXFS', 350000.0),
            'PSAVERT': defaults.get('PSAVERT', 6.0)
        }
        
        X_sample = pd.DataFrame([input_data])
        X_sample = X_sample[feature_names] # Align columns perfectly
        
        # Predict Segment
        pred_cluster = int(classifier.predict(X_sample)[0])
        cluster_data = CLUSTER_INFO.get(pred_cluster, {"name": f"Segment {pred_cluster}", "desc": "Unknown segment characteristics.", "color": "#9ca3af"})
        
        # Outputs
        rfm_str = f"📅 Recency: {int(recency)} days | 🔄 Frequency: {int(frequency)} purchases | 💰 Monetary: ${monetary:,.2f}"
        ecom_str = f"👁️ Views: {int(views)} | 🛒 Carts: {int(carts)} | 🛍️ Purchases: {int(purchases)} | 🚪 Sessions: {int(sessions)}"
        clv_str = f"🎯 Simulated 12m CLV: ${expected_clv:,.2f} | 📈 Active Prob: {active_prob*100:.1f}% | 🔀 Group: Simulation"
        
        desc_html = f"""
        <div style="padding: 15px; border-left: 5px solid {cluster_data['color']}; background: rgba(255,255,255,0.02); border-radius: 4px;">
            <h3 style="margin: 0; color: {cluster_data['color']}; font-size: 1.3em;">{cluster_data['name']}</h3>
            <p style="margin: 8px 0 0 0; color: #d1d5db; line-height: 1.5;">{cluster_data['desc']}</p>
        </div>
        """
        
        # Plots
        importance_df = classifier.explain_local(X_sample, pred_cluster)
        fig_shap = make_local_shap_plot(importance_df, cluster_data['name'])
        fig_pca = make_pca_scatter_plot(X_sample, pred_cluster, cluster_data['color'])
        
        return rfm_str, ecom_str, clv_str, desc_html, fig_shap, fig_pca
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error running simulation: {e}", "", "", "", None, None

# Get a sample of real Customer IDs for the dropdown selection
if df_baseline is not None:
    sample_ids = df_baseline['customer_id'].head(100).tolist()
else:
    sample_ids = []

# Custom CSS for gorgeous aesthetics and dark-mode styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: radial-gradient(circle at 10% 20%, hsla(230, 45%, 11%, 1) 0%, hsla(240, 60%, 4%, 1) 90.1%) !important;
    color: #f3f4f6 !important;
}

.prose h1 {
    font-weight: 700 !important;
    background: linear-gradient(135deg, #c084fc, #6366f1) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-shadow: 0 4px 20px rgba(99, 102, 241, 0.15) !important;
    text-align: center !important;
}

.gradio-container button.primary {
    background: linear-gradient(135deg, #8b5cf6, #4f46e5) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.gradio-container button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

/* Custom cards for stats */
.stats-box {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    transition: all 0.3s ease !important;
}

.stats-box:hover {
    border-color: rgba(167, 139, 250, 0.25) !important;
    background: rgba(255, 255, 255, 0.03) !important;
}
"""

with gr.Blocks() as demo:
    gr.HTML("<h1 style='font-size: 2.8em; margin-bottom: 5px; text-align: center;'>🛍️ Real-Time Customer Segmentation Dashboard</h1>")
    gr.HTML("<p style='text-align: center; color: #9ca3af; font-size: 1.1em; margin-bottom: 30px;'>Powered by PCA, BIC-Gaussian Mixture Soft Clustering, and Random Forest Classifiers.</p>")
    
    with gr.Tabs():
        # TAB 1: Real Customer Batch Explorer
        with gr.Tab("🔍 Customer Batch Explorer & Lookup"):
            gr.HTML("<p style='color: #a78bfa; font-weight: 500; margin-bottom: 15px;'>Select an active Customer ID from the real ingested cohort (~5.5M customers baseline) to view behavioral metrics and predictions.</p>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    id_dropdown = gr.Dropdown(
                        choices=sample_ids, 
                        value=sample_ids[0] if sample_ids else None, 
                        label="Select Customer ID", 
                        interactive=True
                    )
                    
                    gr.HTML("<div style='margin-top: 20px;'><h4 style='color: #9ca3af; margin-bottom: 10px;'>📊 Predictor Outputs</h4></div>")
                    lookup_rfm = gr.Textbox(label="Transactional RFM Profile", interactive=False)
                    lookup_ecom = gr.Textbox(label="eCommerce Activity Metrics", interactive=False)
                    lookup_clv = gr.Textbox(label="CLV & A/B Testing Routing", interactive=False)
                    
                    gr.HTML("<div style='margin-top: 20px;'><h4 style='color: #9ca3af; margin-bottom: 10px;'>🏷️ Segment Classification</h4></div>")
                    segment_desc = gr.HTML(label="Predicted Segment Description")
                    
                    lookup_btn = gr.Button("Query Customer", variant="primary")
                    
                with gr.Column(scale=2):
                    with gr.Row():
                        plot_shap = gr.Plot(label="Local Feature Contribution (Random Forest Gini)")
                        plot_pca = gr.Plot(label="Placement in PCA Latent Space")
            
            lookup_btn.click(
                fn=process_lookup,
                inputs=[id_dropdown],
                outputs=[lookup_rfm, lookup_ecom, lookup_clv, segment_desc, plot_shap, plot_pca]
            )
            
        # TAB 2: Interactive Simulator
        with gr.Tab("🧪 Segment Simulator & Scenario Sandbox"):
            gr.HTML("<p style='color: #a78bfa; font-weight: 500; margin-bottom: 15px;'>Simulate a custom consumer's behaviour by moving sliders across Transactional, eCommerce, and routine grocery indicators.</p>")
            
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.HTML("<h3 style='color: #c084fc; margin-top: 0;'>1. Transactional Metrics</h3>")
                    sim_recency = gr.Slider(minimum=1, maximum=365, value=45, label="Recency (Days since last purchase)")
                    sim_frequency = gr.Slider(minimum=0, maximum=50, value=5, label="Frequency (Number of repeat transactions)")
                    sim_monetary = gr.Slider(minimum=5, maximum=1000, value=75, label="Monetary Value ($ average per order)")
                    
                    gr.HTML("<h3 style='color: #c084fc; margin-top: 15px;'>2. eCommerce Activity</h3>")
                    sim_views = gr.Slider(minimum=0, maximum=1000, value=120, label="Total Session Views")
                    sim_carts = gr.Slider(minimum=0, maximum=100, value=12, label="Total Items Added to Cart")
                    sim_purchases = gr.Slider(minimum=0, maximum=50, value=6, label="Total Completed eCommerce Purchases")
                    sim_sessions = gr.Slider(minimum=1, maximum=200, value=15, label="Total Session Volume")
                    sim_abandon = gr.Dropdown(choices=["No", "Yes"], value="No", label="Abandoned Checkout Flag")
                    
                    gr.HTML("<h3 style='color: #c084fc; margin-top: 15px;'>3. Instacart / Grocery Habits</h3>")
                    sim_instacart = gr.Slider(minimum=0, maximum=100, value=10, label="Instacart Total Order Volume")
                    sim_days = gr.Slider(minimum=0, maximum=30, value=7, label="Average Days Between Grocery Orders")
                    
                    gr.HTML("<h3 style='color: #c084fc; margin-top: 15px;'>4. Environmental Context</h3>")
                    sim_precip = gr.Dropdown(choices=["No", "Yes"], value="No", label="Purchased During Precipitation (Rain/Snow)")
                    
                    sim_btn = gr.Button("Simulate Customer Segment", variant="primary")
                    
                with gr.Column(scale=2):
                    gr.HTML("<h3 style='color: #a78bfa; margin-top: 0;'>🔮 Simulation Predictions</h3>")
                    sim_out_rfm = gr.Textbox(label="Transactional RFM Profile", interactive=False)
                    sim_out_ecom = gr.Textbox(label="eCommerce Activity Metrics", interactive=False)
                    sim_out_clv = gr.Textbox(label="CLV & A/B Testing Routing", interactive=False)
                    
                    sim_out_desc = gr.HTML(label="Predicted Segment Description")
                    
                    with gr.Row():
                        sim_plot_shap = gr.Plot(label="Local Feature Contribution (Random Forest Gini)")
                    with gr.Row():
                        sim_plot_pca = gr.Plot(label="Placement in PCA Latent Space")
                        
            sim_btn.click(
                fn=process_simulation,
                inputs=[
                    sim_recency, sim_frequency, sim_monetary, 
                    sim_views, sim_carts, sim_purchases, sim_sessions, sim_abandon, 
                    sim_instacart, sim_days, sim_precip
                ],
                outputs=[sim_out_rfm, sim_out_ecom, sim_out_clv, sim_out_desc, sim_plot_shap, sim_plot_pca]
            )
            
        # TAB 3: About & Metrics
        with gr.Tab("ℹ️ About & Metrics Explanation"):
            gr.Markdown("""
            ### 🛒 Project Overview
            This application predicts Customer Segments and their Lifetime Value (CLV) using several machine learning and statistical models on real-world grocery, eCommerce, and mall purchase data.
            
            We use **PCA** for Dimensionality Reduction, **Gaussian Mixture Models (GMM)** with BIC optimization for soft clustering, **BG/NBD** and **Gamma-Gamma** models for CLV, and **Random Forest** for segment prediction and feature importance explainability.
            
            ---
            ### 📊 Explanation of Metrics
            
            #### 1. Transactional RFM Metrics
            * **Recency:** The number of days since the customer's last purchase.
            * **Frequency:** The total number of repeat transactions made by the customer.
            * **Monetary Value:** The average amount spent per order.
            
            #### 2. eCommerce    Activity
            * **Views:** Total number of product pages visited.
            * **Carts:** Total number of items added to the shopping cart.
            * **Purchases:** Successful eCommerce transactions completed.
            * **Sessions:** Number of independent browsing sessions.
            * **Abandoned Checkout Flag:** Whether the user has an abnormally high ratio of adding items to the cart without completing the purchase.
            
            #### 3. Instacart / Grocery Habits
            * **Total Orders:** Number of grocery orders placed on Instacart.
            * **Days Between Orders:** The average gap (in days) between successive grocery runs.
            
            #### 4. Macro-Economic & Environmental Context
            * **Precipitation Flag:** Did the customer make purchases specifically during rain or snow? (Sourced from Open-Meteo API).
            * **CPIAUCSL (Inflation):** Consumer Price Index for All Urban Consumers (Sourced from FRED API).
            * **RSXFS (Retail Sales):** Advance Retail Sales: Retail Trade (Sourced from FRED API).
            * **PSAVERT (Savings Rate):** Personal Saving Rate (Sourced from FRED API).
            
            #### 5. Output Predictions
            * **Expected 12m CLV:** The predicted monetary value the customer will generate for the business over the next 12 months.
            * **Active Probability:** The mathematical likelihood that the customer hasn't permanently churned given their recency and frequency trends (from the BG/NBD model).
            * **A/B Testing Routing:** Shows whether this user should be in a control or variant group for targeted marketing campaigns based on their risk profile.
            """)
            
    # Load initial default values on app launch
    if sample_ids:
        demo.load(
            fn=process_lookup,
            inputs=[id_dropdown],
            outputs=[lookup_rfm, lookup_ecom, lookup_clv, segment_desc, plot_shap, plot_pca]
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css)
