# Market Segmentation Analysis via K-Means Clustering: A Data-Driven Approach to Retail Customer Profiling

**Abstract**
Customer segmentation is an essential analytical process for modern retail environments seeking to optimize marketing efficiency and resource allocation. This project employs unsupervised machine learning—specifically K-Means clustering—to autonomously partition a dataset of mall customers into distinct behavioral profiles based on demographic and financial features. Through comprehensive Exploratory Data Analysis (EDA) and rigorous statistical cluster validation, we identified six predominant customer segments. To operationalize these findings, an interactive Graphical User Interface (GUI) was developed, allowing non-technical stakeholders to perform instantaneous predictive profiling of new consumers.

---

## 1. Introduction & Problem Statement
Understanding consumer behavior is a pivotal challenge in the retail industry. In the absence of targeted profiling, marketing campaigns remain generalized, often resulting in inefficient budget expenditure and suboptimal conversion rates. 

The primary objective of this project is to analyze a dataset of mall customers to group them into mutually exclusive segments. By identifying these target clusters, marketing teams can formulate precise, data-driven strategies tailored to specific demographics, ultimately maximizing both profitability and customer satisfaction.

## 2. Dataset Description
The empirical data utilized in this study was sourced from Kaggle:
[Mall Customers Dataset](https://www.kaggle.com/datasets/shwetabh123/mall-customers)

The dataset encompasses 200 records, containing the following features for each customer:
- **CustomerID**: Unique identifier.
- **Gender**: Categorical demographic identifier (Male/Female).
- **Age**: Numerical age in years.
- **Annual Income (k$)**: Quantitative measure of the customer's yearly income in thousands of dollars.
- **Spending Score (1-100)**: A proprietary metric assigned by the mall based on customer purchasing behavior and expenditure history.

## 3. Exploratory Data Analysis (EDA)
Prior to model implementation, comprehensive Exploratory Data Analysis (EDA) was conducted to ascertain the underlying statistical distributions of the dataset.

### 3.1 Gender Distribution
The dataset exhibits a slight skew towards female customers. We visualized this using both absolute counts (barplot) and proportional representations (pie chart).
![Gender Distribution](Customer_Demographics_Gender_Distribution.png)

### 3.2 Age Demographics
To understand generational purchasing power, the age distribution was plotted utilizing a histogram with a Kernel Density Estimate (KDE) and an accompanying boxplot. The distribution indicates a higher concentration of customers in their late twenties to early thirties.
![Age Distribution](Customer_Demographics_Age_Distribution_Analysis.png)

### 3.3 Financial Metrics: Annual Income
The financial standing of the mall's customer base follows a normal distribution curve centered around the $50k-$75k range, as evidenced by the density plot.
![Annual Income Distribution](Customer_Financials_Annual_Income_Distribution.png)

---

## 4. Methodology: K-Means Clustering
To partition the dataset autonomously without pre-labeled data, we applied the **K-Means Clustering** algorithm. K-Means operates by iteratively minimizing the multi-dimensional Euclidean distance between data points and their respective cluster centroids.

A critical prerequisite for K-Means is determining the optimal number of clusters ($k$). We utilized two robust mathematical heuristics for this determination:

### 4.1 The Elbow Method
The Elbow Method calculates the Within-Cluster-Sum-of-Squares (WCSS) across varying values of $k$. As $k$ increases, WCSS inherently drops. The optimal $k$ is identified at the inflection point ("elbow") of the curve, representing the threshold of diminishing returns in variance reduction.
![Elbow Method](Cluster_Optimization_Elbow_Method_Evaluation.png)

### 4.2 Average Silhouette Method
To validate the findings of the Elbow Method, we computed the Average Silhouette Score. This metric quantifies intra-cluster cohesion versus inter-cluster separation. Our analysis maximized the silhouette coefficient at **k = 6**, confirming it as the mathematically optimal number of clusters.
![Silhouette Method](Cluster_Optimization_Average_Silhouette_Score_Analysis.png)

---

## 5. Results & Segmentation Analysis
Applying K-Means with $k=6$ successfully partitioned the dataset into six distinct behavioral profiles. The clusters can be interpreted as follows:
1. **Average Income, Average Spending** (Older Demographics)
2. **Low Income, Low Spending** (Sensible Spenders)
3. **Average Income, Average Spending** (Younger Demographics)
4. **Low Income, High Spending** (Careless Spenders)
5. **High Income, High Spending** (Prime Target Customers)
6. **High Income, Low Spending** (Careful Spenders)

### 5.1 Bivariate Visualizations
Plotting Annual Income against Spending Score clearly delineates the distinct centroids formed by the algorithm.
![Income vs Spending](Customer_Segmentation_Income_vs_Spending_Score_Clusters.png)

Visualizing Age against Spending Score reveals secondary patterns, particularly highlighting how the "High Spending" clusters tend to be populated by younger demographics.
![Age vs Spending](Customer_Segmentation_Age_vs_Spending_Score_Clusters.png)

### 5.2 Dimensionality Reduction (PCA)
Because clustering was executed across three dimensions simultaneously (Age, Income, Spending Score), we applied Principal Component Analysis (PCA) to map the multi-dimensional variance onto a 2D plane for clearer mathematical visualization.
![PCA View](Customer_Segmentation_2D_PCA_Dimensionality_Reduction.png)

---

## 6. Application Integration: Desktop Graphical User Interface (GUI)
To ensure the analytical model is actionable for retail management and non-technical stakeholders, a standalone Desktop GUI was engineered using `tkinter` and `matplotlib`.

![GUI Application Dashboard](guiAppDash.png)

### 6.1 Application Mechanics
The application abstracts the underlying Python codebase and securely loads the exported machine learning model (`kmeans_model.pkl`). 

#### User Inputs:
- **Age (Years)**: The numeric age of the consumer.
- **Annual Income (k$)**: The estimated yearly income of the consumer, inputted in thousands (e.g., entering '60' equates to $60,000).
- **Spending Score (1-100)**: The designated proprietary score ranging from 1 to 100.

#### Output and Visualization:
Upon executing the "Predict Segment" command, the application sanitizes the inputs, performs inference through the K-Means model, and instantly classifies the consumer into one of the six established profiles. 

Furthermore, the right-hand pane renders an interactive scatter plot of the original dataset. A prominent **red star marker** is dynamically superimposed onto the graph to visually demonstrate exactly where the new consumer is positioned relative to the overall market population.

## 7. Conclusion
This project successfully demonstrates the utility of unsupervised machine learning in transforming raw transactional and demographic data into actionable business intelligence. By integrating the highly optimized K-Means model into an intuitive graphical interface, retail strategists can seamlessly profile future customers and execute highly targeted marketing initiatives with surgical precision.

---
**References:**
- Dataset: [Kaggle - Mall Customers](https://www.kaggle.com/datasets/shwetabh123/mall-customers)
