import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
def load_and_preprocess_data(file, handle_missing='median', handle_outliers=True, normalize=True):
    data = pd.read_csv(file)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if handle_missing == 'median':
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    elif handle_missing == 'mean':
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    elif handle_missing == 'drop':
        data = data.dropna()

    if handle_outliers:
        z_scores = np.abs(zscore(data[numeric_cols]))
        data = data[(z_scores < 3).all(axis=1)]

    if normalize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[numeric_cols])
        return pd.DataFrame(data_scaled, columns=numeric_cols)
    else:
        return data[numeric_cols]

# ---------------------------
# 2. Clustering Algorithms
# ---------------------------

def apply_kmeans(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_

def apply_kmedoids(X, n_clusters):
    model = KMedoids(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_

def apply_agnes(X, n_clusters, linkage_method='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    return labels

def apply_diana(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    labels = model.fit_predict(X)
    return labels

def apply_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels

# ---------------------------
# 3. Visualization Functions
# ---------------------------

def plot_scatter(X, labels, centers=None, title='Clustering result'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='tab10')
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200)
    plt.title(title)
    st.pyplot(plt)

def plot_elbow(X, max_k=10):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    st.pyplot(plt)

def plot_dendrogram(X, method='ward'):
    Z = linkage(X, method=method)
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Dendrogram')
    st.pyplot(plt)

# ---------------------------
# 4. Metrics Calculation
# ---------------------------

def calculate_metrics(X, labels):
    if len(set(labels)) > 1 and -1 not in labels:
        silhouette = silhouette_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
    else:
        silhouette, davies, calinski = None, None, None

    return silhouette, davies, calinski

# ---------------------------
# 5. Streamlit Interface
# ---------------------------

st.title('Clustering Interface for Data Mining TP4')

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.sidebar.header('Preprocessing Options')
    handle_missing = st.sidebar.selectbox("Handle missing values", ['median', 'mean', 'drop'])
    handle_outliers = st.sidebar.checkbox("Remove outliers (Z-Score)", value=True)
    normalize = st.sidebar.checkbox("Normalize features", value=True)

    X = load_and_preprocess_data(uploaded_file, handle_missing, handle_outliers, normalize)

    st.sidebar.header('Clustering Options')
    method = st.sidebar.selectbox('Choose clustering method', ['KMeans', 'KMedoids', 'AGNES', 'DIANA', 'DBSCAN'])
    if method in ['KMeans', 'KMedoids', 'AGNES', 'DIANA']:
        n_clusters = st.sidebar.slider('Number of clusters', 2, 10, 3)
    if method == 'DBSCAN':
        eps = st.sidebar.slider('EPS (neighborhood size)', 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider('Min samples', 1, 10, 5)

    if st.sidebar.button('Run Clustering'):
        if method == 'KMeans':
            labels, centers = apply_kmeans(X, n_clusters)
            plot_scatter(X.values, labels, centers, 'KMeans Clustering')
        elif method == 'KMedoids':
            labels, centers = apply_kmedoids(X, n_clusters)
            plot_scatter(X.values, labels, centers, 'KMedoids Clustering')
        elif method == 'AGNES':
            labels = apply_agnes(X, n_clusters)
            plot_scatter(X.values, labels, None, 'AGNES Clustering')
            plot_dendrogram(X)
        elif method == 'DIANA':
            labels = apply_diana(X, n_clusters)
            plot_scatter(X.values, labels, None, 'DIANA Clustering')
            plot_dendrogram(X, method='complete')
        elif method == 'DBSCAN':
            labels = apply_dbscan(X, eps, min_samples)
            plot_scatter(X.values, labels, None, 'DBSCAN Clustering')

        silhouette, davies, calinski = calculate_metrics(X, labels)
        metrics_data = {
            'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score'],
            'Value': [silhouette, davies, calinski]
        }
        metrics_df = pd.DataFrame(metrics_data)

        st.subheader('Clustering Metrics Comparison')
        st.dataframe(metrics_df)

        st.subheader('Optional: Elbow Method (for KMeans/KMedoids)')
        if method in ['KMeans', 'KMedoids']:
            plot_elbow(X)