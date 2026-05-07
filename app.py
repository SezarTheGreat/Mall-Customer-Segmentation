import gradio as gr
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and data
try:
    model = joblib.load('kmeans_model.pkl')
    df = pd.read_csv('Mall_Customers.csv')
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    df['Cluster'] = model.predict(X)
    df['Cluster Name'] = df['Cluster'].map(lambda x: f"Cluster {x+1}")
except Exception as e:
    print(f"Error loading model or dataset: {e}")
    model = None
    df = None

CLUSTER_DESCRIPTIONS = {
    0: "Cluster 1: Average Income, Average Spending (Older Demographics)",
    1: "Cluster 2: Low Income, Low Spending (Sensible Spenders)",
    2: "Cluster 3: Average Income, Average Spending (Younger Demographics)",
    3: "Cluster 4: Low Income, High Spending (Careless Spenders)",
    4: "Cluster 5: High Income, High Spending (Prime Target Customers)",
    5: "Cluster 6: High Income, Low Spending (Careful Spenders)"
}

def predict_segment(age, income, score):
    if model is None or df is None:
        return "Error: Model or Data not loaded.", None
        
    try:
        prediction = model.predict([[age, income, score]])
        cluster_idx = prediction[0]
        desc = CLUSTER_DESCRIPTIONS.get(cluster_idx, f"Cluster {cluster_idx + 1}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        cluster_order = sorted(df['Cluster Name'].unique())
        
        sns.scatterplot(
            data=df, 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)', 
            hue='Cluster Name', 
            palette='tab10', 
            hue_order=cluster_order,
            alpha=0.3,
            ax=ax,
            legend=False
        )
        
        ax.scatter([income], [score], color='red', marker='*', s=400, edgecolor='black', linewidth=1.5, label='You / New Consumer', zorder=5)
        ax.set_title(f"Placement on Cluster Map ({desc.split(':')[0]})", fontweight='bold')
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
        return desc, fig
    except Exception as e:
        return str(e), None

# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛍️ Mall Customer Segmentation Predictor")
    gr.Markdown("Enter customer details below to predict their cluster and see where they stand among other customers.")
    
    with gr.Row():
        with gr.Column(scale=1):
            age_input = gr.Number(label="Age (Years)", value=30)
            income_input = gr.Number(label="Annual Income (k$)", value=50)
            score_input = gr.Number(label="Spending Score (1-100)", value=50)
            predict_btn = gr.Button("Predict Segment", variant="primary")
            
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Predicted Segment")
            output_plot = gr.Plot(label="Customer Placement Graph")
            
    predict_btn.click(
        fn=predict_segment, 
        inputs=[age_input, income_input, score_input], 
        outputs=[output_text, output_plot]
    )

if __name__ == "__main__":
    demo.launch()
