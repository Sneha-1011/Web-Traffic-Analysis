import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

# Load the dataset
web_traffic_data = pd.read_csv("Traffic Dataset.csv")

# Pre-processing
web_traffic_data['Date'] = pd.to_datetime(web_traffic_data['Date'])
web_traffic_data.set_index('Date', inplace=True)

print(web_traffic_data.head())  # View the data

# missing values
missing_values = web_traffic_data.isnull().sum()
print("Number of missing values:")
print(missing_values)

# Fill missing values 
web_traffic_data['Rating'].fillna(web_traffic_data['Rating'].mean(), inplace=True)

# Label Encode
label_encoder = LabelEncoder()
web_traffic_data['Main Traffic Source'] = label_encoder.fit_transform(web_traffic_data['Main Traffic Source'])

# category mapping
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
for category, encoded_value in category_mapping.items():
    print(f"Category: {category} - {encoded_value}")

# descriptive
print(web_traffic_data.head())
print(web_traffic_data.describe())


# Exploratory Analysis

desktop_share = web_traffic_data['Desktop Share']
mobile_share = web_traffic_data['Mobile Share']

# Line plot for Desktop and Mobile Shares
plt.plot(web_traffic_data.index, desktop_share, label='Desktop Share', color='blue')
plt.plot(web_traffic_data.index, mobile_share, label='Mobile Share', color='orange')
plt.xlabel('Date')
plt.ylabel('Share')
plt.title('Desktop Share vs Mobile Share')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Rating
plt.fill_between(web_traffic_data.index, web_traffic_data['Rating'], color='skyblue', alpha=0.4)
plt.plot(web_traffic_data.index, web_traffic_data['Rating'], color='blue')
plt.xlabel('Date')
plt.ylabel('Rating')
plt.title('Rating')
plt.xticks(rotation=45)
plt.show()

# Main Traffic Source vs Traffic
traffic_source_labels = label_encoder.inverse_transform(web_traffic_data['Main Traffic Source'].unique())
traffic_source_counts = web_traffic_data['Main Traffic Source'].value_counts().sort_index()
plt.bar(traffic_source_labels, traffic_source_counts)

plt.bar(traffic_source_labels, traffic_source_counts, color=['orange', 'green', 'blue', 'purple'])
plt.xlabel('Main Traffic Source')
plt.ylabel('Traffic')
plt.title('Exploratory Analysis: Main Traffic Source vs Traffic')
plt.xticks(rotation=45)
plt.show()

warnings.filterwarnings("ignore")

# ARIMA

# Split the data 

train_data = web_traffic_data['Traffic']
p, d, q = 1,0,1

# fit the ARIMA model
arima_model = ARIMA(train_data, order=(p, d, q))
arima_model_fit = arima_model.fit()

# predictions
predictions = arima_model_fit.predict(start=152, end=231)
print(predictions)

# Calculate Mean Absolute Error, Squared Error and Error Percentage
test_data = train_data[152:233]  
print(test_data)
mae = mean_absolute_error(test_data, predictions)
print("Mean Absolute Error (MAE):", mae)
mse = mean_squared_error(test_data, predictions)
print("Mean Squared Error (MSE):", mse)
mape = mean_absolute_percentage_error(test_data, predictions)
print("Mean Absolute Percentage Error (MAPE):", mape)

plt.plot(test_data.index, test_data.values, label='Actual')

plt.plot(predictions.index, predictions.values, label='Predicted')

plt.xlabel('Past')
plt.ylabel('Traffic')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()

# PCA and K-means Clustering

X = web_traffic_data[['Traffic', 'Desktop Share', 'Mobile Share', 'Rating', 'Main Traffic Source']]
y = web_traffic_data['Traffic']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_data)
print(X_pca)


scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow')
plt.colorbar(scatter, label='Traffic')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Correlation Matrix
scaled_data = pd.DataFrame(scaled_data, columns=X.columns)
sns.heatmap(scaled_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

components = pca.components_
print("Principal Components:")
for i, component in enumerate(components):
    print(f"Principal Component {i+1}: {component}")

k = 2
kmeans = KMeans(n_clusters=k, n_init=10)
kmeans.fit(X_pca)
colors = ['yellow', 'red']
markers = ['o', 's']

# Scatter plot with cluster colors and markers
for i in range(k):
    plt.scatter(X_pca[kmeans.labels_ == i, 0], X_pca[kmeans.labels_ == i, 1], color=colors[i], marker=markers[i],
                label=f'Cluster {i+1}')

# cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='X',
            label='Cluster Centers')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-means Clustering")
plt.legend()
plt.show()
