---
title: Mall Customer Segmentation
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
tags:
- scikit-learn
- clustering
- random-forest
- gmm
- pca
- lifetimes
- customer-lifetime-value
- explainable-ai
---

# Real-Time Customer Segmentation and Lifetime Value (CLV) Forecasting Pipeline
### *An End-to-End Enterprise Architecture using Dimensionality Reduction (PCA), Gaussian Mixture Soft Clustering (GMM-BIC), Probabilistic Lifetimes Modeling (BG/NBD & Gamma-Gamma), and Explainable Machine Learning (Random Forest)*

---

## 1. Executive Summary & Abstract
In modern multi-channel retail ecosystems, designing personalized customer engagement plans at scale requires a deep understanding of transactional behavior, digital interaction habits, and macro-environmental contexts. Historically, organizations have relied on static, hard-boundary heuristics (such as simple K-Means on basic Recency, Frequency, and Monetary (RFM) metrics) that fail to capture the multi-dimensional, probabilistic nature of customer life cycles. 

This paper introduces a production-ready, mathematically rigorous customer intelligence architecture. The framework ingests high-scale unified datasets across mall customer records, multi-category eCommerce behaviors, and routine grocery orders. It scales and reduces this high-dimensional feature space using Principal Component Analysis (PCA), clusters customers into soft probabilistic segments using a Gaussian Mixture Model (GMM) optimized via the Bayesian Information Criterion (BIC), forecasts individual 12-month Customer Lifetime Value (CLV) using joint BG/NBD and Gamma-Gamma models, and provides local explainability through a supervised Random Forest ensemble. The architecture is completed by an automated statistical monitoring component using a Two-Sample Kolmogorov-Smirnov (K-S) drift test and an adaptive A/B testing routing engine.

---

## 2. Theoretical & Mathematical Formulations

```
                  +------------------------------------------------------+
                  |                RAW UNIFIED INGESTION                 |
                  |     (Mall Customers + eCommerce + Instacart Data)    |
                  +--------------------------+---------------------------+
                                             |
                                             v
                  +------------------------------------------------------+
                  |               DATA ENRICHMENT PIPELINE               |
                  |     (FRED Macroeconomics + Open-Meteo Weather)       |
                  +--------------------------+---------------------------+
                                             |
                  +--------------------------+---------------------------+
                  |                                                      |
                  v                                                      v
  +-------------------------------+                      +-------------------------------+
  |    PROBABILISTIC LIFE CYCLES  |                      |    DIMENSIONALITY REDUCTION   |
  |   (BG/NBD Active Probability) |                      |     (StandardScaler + PCA)    |
  +---------------+---------------+                      +---------------+---------------+
                  |                                                      |
                  v                                                      v
  +-------------------------------+                      +-------------------------------+
  |     MONETARY VALUE MODEL      |                      |    GAUSSIAN MIXTURE MODEL     |
  |    (Gamma-Gamma Fitter)       |                      |    (Soft Clustering & BIC)    |
  +---------------+---------------+                      +---------------+---------------+
                  |                                                      |
                  +--------------------------+---------------------------+
                                             |
                                             v
                  +------------------------------------------------------+
                  |             SUPERVISED CLASSIFICATION &              |
                  |            EXPLAINABLE AI ENGINE (SHAP/RF)           |
                  +--------------------------+---------------------------+
                                             |
                                             v
                  +------------------------------------------------------+
                  |            REAL-TIME A/B TEST ROUTER &               |
                  |          STATISTICAL DRIFT MONITOR (KS TEST)         |
                  +------------------------------------------------------+
```

### A. Customer Life-Cycle Modeling: The BG/NBD Model
The Beta-Geometric/Negative Binomial Distribution (BG/NBD) model describes the transactional behavior of customers who are active for a period, and then permanently churn (go "inactive").

1. **Transaction Process:** While active, the number of transactions made by a customer in a time period of length $t$ follows a Poisson process with transaction rate $\lambda$.
2. **Heterogeneity in $\lambda$:** The transaction rate $\lambda$ varies across customers according to a Gamma distribution with shape parameter $r$ and scale parameter $\alpha$:
   $$f(\lambda | r, \alpha) = \frac{\alpha^r \lambda^{r-1} e^{-\lambda \alpha}}{\Gamma(r)}$$
3. **Churn Process:** After any transaction, a customer becomes permanently inactive with probability $p$. The point of churn follows a Beta-Geometric distribution.
4. **Heterogeneity in $p$:** The dropout probability $p$ varies across customers according to a Beta distribution with parameters $a$ and $b$:
   $$f(p | a, b) = \frac{p^{a-1} (1-p)^{b-1}}{B(a, b)}$$

The probability that a customer with purchase history $(x, t_x, T)$ (where $x$ is frequency, $t_x$ is recency, and $T$ is customer age) is still active at time $T$ is formulated as:
$$P(\text{Active} | x, t_x, T, r, \alpha, a, b) = \frac{1}{1 + \frac{a}{b+x} \left(\frac{\alpha + T}{\alpha + t_x}\right)^{r+x}}$$

### B. Expected Monetary Value: The Gamma-Gamma Model
To forecast the monetary spend of the active customer cohort, we utilize the Gamma-Gamma model. This model assumes that a customer's average transaction value varies around their mean transaction value, and this mean value varies across the customer population.

1. **Transaction Values:** The transaction values $Z_i$ for a given customer are independent and identically distributed Gamma variables with shape parameter $p$ and scale parameter $v$.
2. **Heterogeneity in Mean Spend:** The scale parameter $v$ varies across customers according to a Gamma distribution with shape parameter $q$ and scale parameter $\gamma$:
   $$f(v | q, \gamma) = \frac{\gamma^q v^{q-1} e^{-v \gamma}}{\Gamma(q)}$$

This yields the expected average transaction value $E(M | x, m_x)$ for a customer with average observed order value $m_x$ across $x$ transactions:
$$E(M | x, m_x, p, q, \gamma) = \frac{\gamma p + x m_x}{q + x - 1}$$

Combining the BG/NBD transaction counts and the Gamma-Gamma transaction values, the **Expected 12-Month Customer Lifetime Value (CLV)** is calculated as:
$$\text{CLV}_{12m} = \int_{T}^{T+12} E[\text{Transactions}(t)] \cdot E(M) \cdot e^{-d \cdot t} \, dt$$
where $d$ represents the monthly discount rate.

### C. Dimensionality Reduction: Principal Component Analysis (PCA)
With $D$ highly correlated behavioral features, we utilize PCA to project the features into an orthogonal subspace that preserves $\ge 95\%$ of total variance.

Given a standardized feature matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, we compute the covariance matrix $\mathbf{\Sigma}$:
$$\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{X}^T \mathbf{X}$$

We perform eigendecomposition on $\mathbf{\Sigma}$:
$$\mathbf{\Sigma} \mathbf{V} = \mathbf{V} \mathbf{\Lambda}$$
where $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_D]$ are the orthonormal eigenvectors (Principal Components), and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_D)$ represents the eigenvalues corresponding to the variance explained by each component. 

### D. Probabilistic Clustering: Gaussian Mixture Models (GMM)
Rather than assuming rigid customer boundaries, we model the customer distribution as a mixture of $K$ multivariate Gaussian distributions:
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)$$
where $\pi_k$ is the mixing coefficient for cluster $k$ ($\sum_k \pi_k = 1$), $\boldsymbol{\mu}_k$ is the mean vector, and $\mathbf{\Sigma}_k$ is the covariance matrix.

To identify the optimal number of clusters $K$, the pipeline minimizes the **Bayesian Information Criterion (BIC)** to prevent overfitting:
$$\text{BIC} = -2 \ln(\hat{L}) + k_{params} \ln(N)$$
where $\hat{L}$ is the maximized likelihood of the model, $k_{params}$ is the number of estimated parameters, and $N$ is the number of data points.

### E. Supervised Explainability: Random Forest Gini Classifier
To explain the clusters in terms of raw physical units (days, dollars, clicks), we treat the GMM assignments as ground-truth targets and train a **Random Forest Classifier**. Global and local feature importances are calculated via Mean Decrease in Impurity (MDI / Gini Importance):
$$\text{MDI}(X_j) = \frac{1}{M} \sum_{m=1}^M \sum_{t \in T_m : v(t) = X_j} p(t) \Delta i(t)$$
where $M$ is the number of trees, $p(t)$ is the proportion of samples reaching node $t$, and $\Delta i(t)$ is the decrease in Gini impurity achieved by splitting on feature $X_j$.

---

## 3. Recommended Visual Assets for Papers and Documentation
To compile an academic-grade publication or highly professional README, you should generate and embed the following six figures:

### Figure 1: End-to-End Pipeline Architecture Flowchart
*   **Description:** A detailed schematic diagram illustrating data flow.
*   **What it represents:** The unified ingestion of multi-source datasets, macro-economic and precipitation enrichment APIs, split parallel processing for Lifetimes models (probabilistic CLV) and standard scaled PCA reduction, GMM-BIC clustering, and the subsequent explainable Random Forest wrapper outputting custom local/global metrics.
*   **Suggested Implementation:** Render a high-fidelity vector diagram using tool platforms like Draw.io, Lucidchart, or Mermaid.js.

### Figure 2: GMM Cluster Optimization Curve (BIC vs. K)
*   **Description:** A line plot displaying the Bayesian Information Criterion (BIC) score on the y-axis against the number of clusters $K$ (from 2 to 10) on the x-axis.
*   **What it represents:** The mathematical derivation of the optimal cluster threshold. The point where the BIC score reaches a local minimum or exhibits a distinct "elbow" shows the exact number of statistically distinct customer cohorts present in the marketplace.
*   **Suggested Implementation:** Plot using `matplotlib` during the `SoftClusterer` fitting phase.

### Figure 3: 2D Projection of Latent Space (PCA Scatter Plot)
*   **Description:** A scatter plot displaying the first two Principal Components ($PC_1$ and $PC_2$) as axes. 
*   **What it represents:** The distribution of the baseline customer population, colored by their assigned GMM soft cluster. It highlights how the multi-dimensional behavioral coordinates are compressed into distinct, highly segregated groups. In production, overlaying a star-marker representing a new query customer demonstrates the real-time classification mapping.
*   **Suggested Implementation:** Save the graphic output from the PCA tab in the Gradio dashboard interface.

### Figure 4: Global and Local Feature Importance Charts
*   **Description:** A horizontal bar chart illustrating the Gini importances or SHAP-proxy values of individual variables (Recency, Frequency, Monetary, Views, Carts, Savings Rate).
*   **What it represents:** The underlying drivers behind segment assignment. Global feature importance displays the overall predictive weight of features, while local importance displays why a specific customer was assigned to their segment.
*   **Suggested Implementation:** Export the interactive SHAP-proxy plot directly from the Gradio interface.

### Figure 5: Expected 12m CLV vs. Active Probability (BG/NBD) Scatter Distribution
*   **Description:** A density scatter plot mapping Expected 12-Month CLV (y-axis) against Active Probability (x-axis).
*   **What it represents:** The distinction between customer risk profiles and transaction volume. Customers in the top-right represent high-value loyalists, bottom-right are transactional shoppers with low individual margins, top-left are high-value customers at immediate risk of churning (critical intervention targets), and bottom-left represent fully inactive/churned profiles.
*   **Suggested Implementation:** Build using `seaborn.jointplot` or `plotly.express.scatter`.

### Figure 6: A/B Testing Routing Engine & Cohort Allocation Flow Chart
*   **Description:** A directional flowchart mapping incoming customers to targeted treatment cohorts.
*   **What it represents:** The business implementation logic. It illustrates how the real-time routing engine takes GMM classifications, CLV forecasts, and active probabilities, filters them through eligibility rules, and splits them into control and variant groups for hyper-personalized marketing.
*   **Suggested Implementation:** Construct using a professional diagrams application.

---

## 4. Engineering & Ingestion Pipeline Details

### Unified Multi-Source Ingestion
The ingestion engine (`ingestion.py`) stitches together three distinct retail behavior models:
1.  **Mall Customer Demographic Data:** Focuses on core customer dimensions (Age, Gender, Annual Income, Spending Score).
2.  **eCommerce Digital Activity:** Tracks high-resolution session mechanics (total views, cart additions, purchase conversions, abandoned checkouts).
3.  **Routine Grocery Ordering (Instacart Model):** Models replenishment frequency, average days between orders, and cumulative grocery spend.

### Contextual API Enrichment
To prevent data isolation, the enrichment module (`enrichment.py`) pulls contextual data from external endpoints:
*   **Macro-Economic Indicators (FRED API):** Pulls live indicators for inflation (`CPIAUCSL`), retail sales indices (`RSXFS`), and personal saving rates (`PSAVERT`) to capture systemic consumer shifts.
*   **Geological & Environmental Context (Open-Meteo API):** Ingests historical and real-time precipitation records based on checkout times to flag rain/snow-influenced shopping behaviors.

### Production Drift Monitoring (Kolmogorov-Smirnov Test)
To protect against model degradation in production, the pipeline incorporates a distribution drift monitor (`monitoring_routing.py`). It regularly compares incoming inference data against the baseline training distribution using a **Two-Sample Kolmogorov-Smirnov (K-S) test**:
$$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
If the computed p-value drops below the threshold (e.g., $\alpha = 0.05$), it flags statistical drift and triggers an automated pipeline retraining request.

---

## 5. Local Execution, Setup & Deployment

### Prerequisite Environment setup
Ensure you have Python installed. Clone the repository and install the standard dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### Running the End-to-End Ingestion & Training Pipeline
To run the full pipeline (ingest raw records, fetch live API variables, run standard scaler and PCA, fit lifetimes models, fit GMM, and serialize pipelines):
```bash
python pipeline.py
```
This script exports the following binary assets to disk:
*   `pca_pipeline.pkl`: The StandardScaler + PCA dimensional mapping.
*   `gmm_model.pkl`: The Gaussian Mixture Model clustering parameters.
*   `clv_bgf_params.pkl`: Fitted parameters ($r, \alpha, a, b$) for transaction forecasting.
*   `clv_ggf_params.pkl`: Fitted parameters ($p, q, v$) for monetary estimation.
*   `xgb_classifier.pkl`: The supervised Random Forest segment predictor wrapper.
*   `feature_names.json`: Schema definitions.
*   `baseline_training_data.parquet`: Baseline cohort dataset.

### Launching the Gradio Web Application
Once the pipeline has completed training and saved the models, launch the premium dark-themed Gradio dashboard:
```bash
python app.py
```
Open **http://127.0.0.1:7860/** in your web browser to explore real customer records or test simulated scenarios.

---

## 6. Target Customer Segment Classifications
The Gaussian Mixture Model maps customers into nine distinct GMM-BIC optimized clusters:

| Cluster | Segment Name | Behavioral Profile & Characteristics | Retention Action & Routing Strategy |
|:---:|:---|:---|:---|
| **0** | **VIP / High-Value Loyalists** | Extremely high 12m CLV, frequent transactions, stable savings indices, low rain sensitivity. | Route to exclusive loyalty program, premium early-access channels. |
| **1** | **Inactive / Churned Customers** | High recency (days since purchase), low active probability, near-zero session views. | Re-engagement email flows, win-back discounts. |
| **2** | **Instacart Bulk Grocery Buyers** | High grocery order volumes, low days between orders, highly consistent replenishment cycles. | Automated subscription options, bulk purchase discounts. |
| **3** | **Window Shoppers** | High session views and cart additions, but low completed purchases. High cart abandonment rates. | High-incentive couponing, targeted cart-recovery emails. |
| **4** | **Weather-Sensitive Impulsives** | Transaction events strongly correlated with high precipitation indices. | Rain-day personalized app push notifications. |
| **5** | **Casual One-Time Retail Buyers** | Moderate average transaction size, but very low frequency. | Introduce post-purchase multi-stage discount cycles. |
| **6** | **Sensible / Standard Spenders** | Balanced browsing habits, moderate income-to-spend ratios. | Maintain standard promotion cycle, suggest matching cross-sells. |
| **7** | **High-Intent Cart Abandoners** | Moderate sessions, high cart additions, high abandoned checkout flag. | Target with immediate free-shipping offers or time-limited vouchers. |
| **8** | **New / Low-Engagement Registrations** | Low recency, minimal overall transaction logs, neutral macro-economic influence. | Introduce welcome series onboarding, low-barrier first-purchase discounts. |