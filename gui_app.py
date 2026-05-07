import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings('ignore')

# Attempt to load the trained K-Means model and dataset
try:
    model = joblib.load('kmeans_model.pkl')
    df = pd.read_csv('Mall_Customers.csv')
    
    # Pre-calculate clusters for the background data
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    df['Cluster'] = model.predict(X)
    df['Cluster Name'] = df['Cluster'].map(lambda x: f"Cluster {x+1}")
except Exception as e:
    print(f"Error loading model or dataset: {e}")
    model = None
    df = None

# Cluster descriptions based on our model's trained centers
CLUSTER_DESCRIPTIONS = {
    0: "Cluster 1: Average Income, Average Spending\n(Older Demographics)",
    1: "Cluster 2: Low Income, Low Spending\n(Sensible Spenders)",
    2: "Cluster 3: Average Income, Average Spending\n(Younger Demographics)",
    3: "Cluster 4: Low Income, High Spending\n(Careless Spenders)",
    4: "Cluster 5: High Income, High Spending\n(Prime Target Customers)",
    5: "Cluster 6: High Income, Low Spending\n(Careful Spenders)"
}

def predict_segment():
    if model is None or df is None:
        messagebox.showerror("Error", "Model or dataset not loaded correctly.")
        return

    try:
        # Fetch inputs
        age_str = age_var.get().strip()
        income_str = income_var.get().strip()
        score_str = score_var.get().strip()
        
        # Check empty
        if not age_str or not income_str or not score_str:
            raise ValueError("All fields are required.")

        age = float(age_str)
        income = float(income_str)
        score = float(score_str)
        
        # Validate ranges
        if not (0 < age < 120):
            raise ValueError("Age must be a valid number between 1 and 120.")
        if not (0 <= income <= 1000):
            raise ValueError("Annual Income is expected in thousands (k$).\nFor example, if the income is $50,000, please just enter '50'.")
        if not (1 <= score <= 100):
            raise ValueError("Spending Score must be between 1 and 100.")
            
        # Predict using the model
        prediction = model.predict([[age, income, score]])
        cluster_idx = prediction[0]
        
        # Update result UI
        result_text.set(CLUSTER_DESCRIPTIONS.get(cluster_idx, f"Cluster {cluster_idx + 1}"))
        
        # Update Graph
        ax.clear()
        
        # Sort cluster names for consistent legend
        cluster_order = sorted(df['Cluster Name'].unique())
        
        sns.scatterplot(
            data=df, 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)', 
            hue='Cluster Name', 
            palette='tab10', 
            hue_order=cluster_order,
            alpha=0.3, # make background points lighter
            ax=ax,
            legend=False
        )
        
        # Plot the exact placement of the new consumer
        ax.scatter([income], [score], color='red', marker='*', s=400, edgecolor='black', linewidth=1.5, label='You / New Consumer', zorder=5)
        
        ax.set_title(f"Placement on Cluster Map ({CLUSTER_DESCRIPTIONS[cluster_idx].split(':')[0]})", fontweight='bold')
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
        canvas.draw()
        
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# Setup the main window
root = tk.Tk()
root.title("Mall Customer Segmentation Predictor")
root.geometry("1000x550")  # Expanded to fit the graph
root.resizable(True, True) # Made resizable

# Try to use a modern theme if available
style = ttk.Style()
if 'clam' in style.theme_names():
    style.theme_use('clam')

# Configure styles
style.configure('TLabel', font=('Segoe UI', 11))
style.configure('TButton', font=('Segoe UI', 12, 'bold'), padding=10)
style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#2c3e50')
style.configure('Result.TLabel', font=('Segoe UI', 13, 'bold'), foreground='#27ae60', justify='center')

# Variables
age_var = tk.StringVar()
income_var = tk.StringVar()
score_var = tk.StringVar()
result_text = tk.StringVar()
result_text.set("Awaiting input...")

# Layout: Left frame for inputs, Right frame for graph
left_frame = ttk.Frame(root)
left_frame.pack(side='left', fill='y', padx=20, pady=20)

right_frame = ttk.Frame(root)
right_frame.pack(side='right', fill='both', expand=True, padx=20, pady=20)

# ----------------- LEFT FRAME (INPUTS) -----------------
ttk.Label(left_frame, text="Customer Segment Predictor", style='Title.TLabel').pack(pady=(0, 20))

# Create input frame
input_frame = ttk.Frame(left_frame)
input_frame.pack(fill='x', padx=10)

# Age Input
ttk.Label(input_frame, text="Age (Years):").grid(row=0, column=0, sticky='w', pady=10)
age_entry = ttk.Entry(input_frame, textvariable=age_var, font=('Segoe UI', 11), width=15)
age_entry.grid(row=0, column=1, pady=10, padx=10, sticky='e')

# Annual Income Input
ttk.Label(input_frame, text="Annual Income (k$):").grid(row=1, column=0, sticky='w', pady=10)
income_entry = ttk.Entry(input_frame, textvariable=income_var, font=('Segoe UI', 11), width=15)
income_entry.grid(row=1, column=1, pady=10, padx=10, sticky='e')

# Spending Score Input
ttk.Label(input_frame, text="Spending Score (1-100):").grid(row=2, column=0, sticky='w', pady=10)
score_entry = ttk.Entry(input_frame, textvariable=score_var, font=('Segoe UI', 11), width=15)
score_entry.grid(row=2, column=1, pady=10, padx=10, sticky='e')

# Predict Button
predict_btn = ttk.Button(left_frame, text="Predict Segment", command=predict_segment, cursor="hand2")
predict_btn.pack(pady=25, fill='x', padx=10)

# Result Display
result_frame = ttk.LabelFrame(left_frame, text="Prediction Output")
result_frame.pack(fill='both', expand=True, padx=10, pady=5)

ttk.Label(result_frame, textvariable=result_text, style='Result.TLabel', wraplength=350).pack(expand=True, pady=15)

# ----------------- RIGHT FRAME (GRAPH) -----------------
fig = Figure(figsize=(6, 5), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Customer Segmentation Placement", fontweight='bold')
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")

# Initial plot state
if df is not None:
    # Just plot the raw points as a gray background before any prediction
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', color='lightgray', alpha=0.5, ax=ax)
    ax.grid(True, linestyle='--', alpha=0.5)

canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill='both', expand=True)

# Focus the first entry
age_entry.focus()

# Start the application
root.mainloop()
