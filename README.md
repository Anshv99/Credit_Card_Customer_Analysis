# Credit_Card_Customer_Analysis

### Overview
This project aims to conduct a comprehensive analysis of credit card customers using the provided dataset from Kaggle. The analysis encompasses data manipulation, visualization, customer segmentation, and predictive modeling to gain insights into customer behavior and predict churn. The project also utilizes Git for version control.

### How to Run the Code
1. Clone this repository to your local machine.
2. Download the dataset from the provided Kaggle link and place it in the data folder.
3. Open Jupyter notebooks in the notebooks folder and open main.ipynb code execution.

### Summary of Findings and Insights

### Step 1: Data Exploration and Preprocessing
- No missing values or repeated entries were found in the dataset.
- 'CLIENTNUM' column was removed as it didn't contribute to clustering analysis.
- Income data was crucial for segmentation, so records with unknown income were excluded.
- Features 'Credit_Limit' and 'Avg_Open_To_Buy' were highly correlated (correlation of 1.0), so only 'Credit_Limit' was retained.

### Step 2: Data Analysis and Visualization
- Demographic analysis revealed insights into age, salary, and marital status distributions.
  #### Age
- This graphs shows Distribution of Customer Age
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/3de1dd21-c249-46a0-bb56-91176cbc6211)

#### Income
- The salary of each client is one of the factors that most impact the services provided by a credit card company to its customers. For this reason, I will choose to exclude records of clients whose income is not known, rather than estimating it, in order to segment them more effectively.
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/103423ae-46a6-4feb-937c-5ad0b5eecba2)

#### Martial Status 
![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/5047b31b-7f1d-4048-ae34-b60411ad71c6)

#### Credit Usage Analysis
- plot a heatmap to show coorelation between the credit card limit, balance, and category and others.
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/4e1450ba-9703-4443-a23c-22d085274883)

### Step 3: Customer Segmentation
- To implement the K-means algorithm we have to preprocess the Data. The feature scaling step is crucial, as K-means clustering is sensitive to the scale of features. It ensures that all features contribute equally to the clustering process, preventing variables with larger scales from dominating the analysis.

  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/57bdab88-2dca-4d91-a43b-3d1d2a16301a)
  
  Filtering only numeric columns for comparison.

- To find the best number of clusters for our data, I'll use two common methods:
1. Elbow Method: I'll plot how the total distance of data points to their assigned clusters changes as we increase the number of clusters. The "elbow" point on the graph suggests the ideal number of clusters, where adding more clusters doesn't significantly reduce the distance.

2. Silhouette Method: I'll calculate how well each data point fits into its assigned cluster compared to other clusters. The average silhouette score measures how similar a point is to its own cluster compared to other clusters. A higher score suggests better-defined clusters. We'll pick the number of clusters with the highest silhouette score.

By combining these methods, we can find the best number of clusters that effectively capture the patterns in our data.

![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/3f0c145b-fdb0-48fa-998a-3dc191b770d4) 

![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/b03efd08-7056-4f97-ac83-464cef7aeba1)

- In the Elbow method graph, we observe it pointing to a number of 3 clusters. Considering that the value of 3 clusters has the best score in the Silhouette method, based on these two indicators, I will define the number of clusters as 3.

  ### Given the high dimensionality of our DataFrame, consisting of 19 columns, I will apply PCA (Principal Component Analysis) to reduce its dimensionality to 2, aiming to simplify the visualization and interpretation of the data in 2D plot.

- After applying PCA we count each cluster

  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/8942454c-a007-45a5-8102-fa432aeef18c)

- plotting the centroids
  
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/59f58787-7196-454a-9746-ab82e1d3e28e)

- plotting the Clusters with datapoints
  
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/b35686d0-4b47-4458-9d51-d9ae65f4126c)

### Step 3.1: Cluster Analysis

- Explore both the numerical and categorical columns, examining the characteristics of each cluster to draw the profile of customers within each group.

#### Numerical Columns
- I will select the main numerical columns and these will be plotted together using pairplots and boxplots.

![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/0e6bfc8b-265a-4d80-b989-a0151c1080d3)

![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/097ac0f6-5a66-40a6-9482-d5b799de9eea)

- In both the pair plot and the box plot, we notice a strong connection between customer age and their tenure with the bank. Cluster 0 comprises older customers with longer bank relationships, while cluster 2 consists of younger customers with shorter tenure. Cluster 1 shows a mix of ages, mostly falling between the medians of clusters 0 and 2, indicating an intermediate age and relationship duration compared to the other clusters. This pattern extends to the duration of the relationship that cluster 1 customers maintain with the bank.

- Regarding total transaction amount, there's little variation across the three clusters, making it difficult to distinguish between them significantly.

- When considering credit limit and average utilization ratio, cluster 1 stands out with a higher credit limit compared to clusters 0 and 2. Despite similar spending patterns observed in the total transaction amount, customers in clusters 0 and 2 exhibit a higher credit utilization ratio due to their lower credit limits. Conversely, customers in cluster 1 have a lower credit utilization ratio due to their higher credit limit.

- In terms of total revolving balance, customers in clusters 0 and 2 carry more outstanding debts on their credit cards compared to those in cluster 1.

#### Categorical Columns
- I'll plot some categorical features that I considered important.
  
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/8866d8bc-4b57-4973-a6b7-d60f93a601fc)
  
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/1f624761-b2ab-4ac6-8b3e-6b23fffa3c72)
  
  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/64166981-73b8-4b54-ba40-c9e4bcbebb0a)


## Churn Model Prediction Preprocessing 

1. Feature-Encoding
Before categorical data can be utilized as input to a machine learning model, it must first be transformed into numerical data. There are multiple ways of handling Categorical variables. The two most widely used techniques:

2. Label Encoding
   
- One-Hot Encoding
- Label Encoding

- A unique integer represents each value of the categorical variable. Issue with label encoding is that it can imply an unintended order to the variable. So, we use it only for ordinal variables of our dataset.
One-Hot Encoding
- It simply creates additional features called dummy variables based on the number of unique values in the categorical feature. Every unique value in the category will be added as a feature. Issue with one-hot encoding is the Dummy Variable Trap - a scenario in which variables are highly correlated to each other - multicollinearity. So, in order to overcome the problem of multicollinearity, one of the dummy variables has to be dropped. We use one-hot encoding for nominal variables of our dataset.

### Feature-Reduction

- Feature reduction techniques can help in identifying and retaining only the most informative features, reducing redundancy. We use different techniques based on the type of the features.

- Continuous features with multi-collinearity - Principal Component Analysis (PCA) Discrete or Categorical features - SelectKBest

 #### Method 1 - Draw the Correlation Matrix to Check for Multi-Collinearity
 
 ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/b5a22281-f1fa-4e17-8455-abe7368d62c7)

#### Method 2 - Calculate Variance Inflation Factor (VIF) to Check for Multi-Collinearity
- The variance inflation factor (VIF) is a measure used to assess multicollinearity in regression analysis. It is also calculated for numerical variables only. The general rule of thumb for interpreting VIF values is as follows

![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/d4f5bf3c-b551-46ac-9730-72fad6c005c7)

- Conducting Pca on Standardized Continous Variables

  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/55373050-0373-4cec-b32a-d39eb8d9ef6f)

- Performing SelectKBest on Discrete and Categorical Variables
- SelectKBest is sensitive to the scale of the features, particularly when using scoring functions that are affected by feature magnitudes, such as the default function for classification, which is the chi-squared test.

  ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/f64aa603-2e25-4076-88b6-39c1d962dd04)


## Building Classification Model

1. KNN Classifier

   ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/d4b2387e-19c5-4038-9ac8-7e0a647558fa)

2.Random Forest Classifer

   ![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/07063795-0ea6-4ebe-8739-c5b3488d6d88)

3.Naive Bayes Classifer

![image](https://github.com/Anshv99/Credit_Card_Customer_Analysis/assets/91983097/0998f288-b6b0-42e2-b7fe-28a6dc93601b)

Here we can see that Naive bayes has the highest accuracy score.

## Contributors
Ansh Vashisht
