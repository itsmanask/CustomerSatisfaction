# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'satisfaction_score': np.random.uniform(1, 5, n_samples),
    'product': np.random.choice(['A', 'B'], n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.randint(20000, 120000, n_samples),
    'other_demographic_feature': np.random.randint(1, 10, n_samples)
}

df = pd.DataFrame(data)

# Display dataset information and head
print("Dataset Information:")
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Perform descriptive statistics
print("\nDescriptive Statistics:")
desc_stats = df.describe()
print(desc_stats)

# Visualize descriptive statistics (Mean and SD) using bar plots
means = desc_stats.loc['mean']
std_devs = desc_stats.loc['std']
features = desc_stats.columns

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
means.plot(kind='bar', color='skyblue')
plt.title("Mean of Each Numerical Feature")
plt.xlabel("Features")
plt.ylabel("Mean")

plt.subplot(1, 2, 2)
std_devs.plot(kind='bar', color='salmon')
plt.title("Standard Deviation of Each Numerical Feature")
plt.xlabel("Features")
plt.ylabel("Standard Deviation")

plt.tight_layout()
plt.show()

# Satisfaction score distribution by product type
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="satisfaction_score", hue="product", kde=True)
plt.title("Satisfaction Scores for Product A and Product B")
plt.xlabel("Satisfaction Score")
plt.ylabel("Frequency")
plt.show()

# Two-sample t-test for satisfaction scores of Product A and Product B
product_a_scores = df[df['product'] == 'A']['satisfaction_score']
product_b_scores = df[df['product'] == 'B']['satisfaction_score']
t_stat, p_value = ttest_ind(product_a_scores, product_b_scores, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
if p_value < 0.05:
    print("There is a significant difference in mean satisfaction scores between Product A and Product B.")
else:
    print("There is no significant difference in mean satisfaction scores between Product A and Product B.")

# Visualize t-test results with density plot and name it "T Testing"
plt.figure(figsize=(10, 5))
sns.kdeplot(product_a_scores, label="Product A", fill=True, color="blue")  # Updated fill parameter
sns.kdeplot(product_b_scores, label="Product B", fill=True, color="orange")  # Updated fill parameter
plt.title("T Testing: Density Plot of Satisfaction Scores for Product A and Product B")
plt.xlabel("Satisfaction Score")
plt.ylabel("Density")
plt.legend()
plt.show()

# Correlation heatmap for numerical features and name it "Correlation"
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots for correlation between demographic features and satisfaction score
demographic_features = ['age', 'income', 'other_demographic_feature']
for feature in demographic_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature, y="satisfaction_score", alpha=0.6)
    plt.title(f"Correlation: {feature.capitalize()} vs. Satisfaction Score")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Satisfaction Score")
    plt.axhline(y=df['satisfaction_score'].mean(), color='r', linestyle='--', label='Mean Satisfaction Score')
    plt.legend()
    plt.show()

# Regression plots for each demographic feature vs satisfaction score and name them "Regression"
for feature in demographic_features:
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=feature, y="satisfaction_score", line_kws={"color": "red"})
    plt.title(f"Regression: {feature.capitalize()} vs. Satisfaction Score")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Satisfaction Score")
    plt.show()

# Prepare data for classification (binary outcome: satisfied/unsatisfied)
threshold = df['satisfaction_score'].median()
df['satisfied'] = df['satisfaction_score'].apply(lambda x: 1 if x >= threshold else 0)
X = df[['age', 'income', 'other_demographic_feature']]
y = df['satisfied']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Classification
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))

# Decision Tree Classification
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

# Feature Reduction using PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize before PCA
pca = PCA(n_components=2)  # Adjust components as needed
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio by PCA components:", pca.explained_variance_ratio_)
print("Total Explained Variance:", sum(pca.explained_variance_ratio_))

# Visualize PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Customer Demographics")
plt.colorbar(label="Satisfied")
plt.show()
