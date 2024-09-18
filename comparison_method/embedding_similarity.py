from comparison_method.base import ComparisonMethodBase
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import cdist  # Import cdist from scipy

class EmbeddingSimilarity(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "Embedding Similarity"

    @staticmethod
    def get_parameter_info(data):
        columns = list(data.columns)
        parameter_info = []

        # Parameters for selecting columns of each data type
        parameter_info.append({
            'name': 'text_columns',
            'type': 'multi_select',
            'options': [col for col in columns if data[col].dtype == 'object'],
            'default': [],
            'description': 'Select text columns for embedding'
        })

        parameter_info.append({
            'name': 'numeric_columns',
            'type': 'multi_select',
            'options': [col for col in columns if np.issubdtype(data[col].dtype, np.number)],
            'default': [],
            'description': 'Select numeric columns to include'
        })

        # Weights for each selected column
        all_selected_columns = columns
        default_weights = {col: 1.0 for col in all_selected_columns}
        parameter_info.append({
            'name': 'weights',
            'type': 'weights',
            'default': default_weights,
            'columns': all_selected_columns,
            'description': 'Weights for each column (0 to 1)'
        })

        # Clustering algorithms
        parameter_info.append({
            'name': 'clustering_algorithm',
            'type': 'select',
            'options': ['None', 'KMeans', 'AgglomerativeClustering', 'DBSCAN', 'OPTICS'],
            'default': 'None',
            'description': 'Clustering algorithm to use'
        })

        # Distance metrics
        parameter_info.append({
            'name': 'distance_metric',
            'type': 'select',
            'options': ['cosine', 'euclidean', 'manhattan', 'chebyshev'],
            'default': 'cosine',
            'description': 'Distance metric for similarity calculation'
        })

        # Number of clusters (for KMeans and AgglomerativeClustering)
        parameter_info.append({
            'name': 'num_clusters',
            'type': int,
            'default': 5,
            'description': 'Number of clusters (for KMeans and AgglomerativeClustering)'
        })

        return parameter_info

    @st.cache_resource
    def load_model(_self):
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_data
    def compute_embeddings(_self, texts):
        return _self.model.encode(texts)

    def __init__(self, **params):
        self.text_columns = params.get('text_columns', [])
        self.numeric_columns = params.get('numeric_columns', [])
        self.weights = params.get('weights', {})
        self.clustering_algorithm = params.get('clustering_algorithm', 'None')
        self.distance_metric = params.get('distance_metric', 'cosine')
        self.num_clusters = params.get('num_clusters', 5)
        self.params = params

        # Initialize the sentence transformer model
        self.model = self.load_model()

    def compute_distance_matrix(self, embeddings):
        if self.distance_metric == 'cosine':
            # Normalize embeddings
            embeddings = normalize(embeddings)
            distance_matrix = cosine_distances(embeddings)
        elif self.distance_metric == 'euclidean':
            distance_matrix = euclidean_distances(embeddings)
        elif self.distance_metric == 'manhattan':
            distance_matrix = manhattan_distances(embeddings)
        elif self.distance_metric == 'chebyshev':
            distance_matrix = cdist(embeddings, embeddings, metric='chebyshev')
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        return distance_matrix


    def compare(self, data, bdv_column):
        # Check if bdv_column exists
        if bdv_column not in data.columns:
            raise ValueError(f"Column '{bdv_column}' not found in data.")

        # Prepare embeddings for text columns
        text_embeddings = []
        if self.text_columns:
            for col in self.text_columns:
                texts = data[col].astype(str).fillna('').tolist()
                embeddings = self.compute_embeddings(texts)
                embeddings *= self.weights.get(col, 1.0)
                text_embeddings.append(embeddings)
            text_embeddings = np.concatenate(text_embeddings, axis=1)
        else:
            text_embeddings = np.array([])

        # Prepare numerical data
        numeric_data = []
        if self.numeric_columns:
            scaler = StandardScaler()
            numeric_values = data[self.numeric_columns].fillna(0).values
            numeric_values = scaler.fit_transform(numeric_values)
            for idx, col in enumerate(self.numeric_columns):
                numeric_values[:, idx] *= self.weights.get(col, 1.0)
            numeric_data = numeric_values
        else:
            numeric_data = np.array([])

        # Combine embeddings and numeric data
        if text_embeddings.size and numeric_data.size:
            combined_embeddings = np.hstack((text_embeddings, numeric_data))
        elif text_embeddings.size:
            combined_embeddings = text_embeddings
        elif numeric_data.size:
            combined_embeddings = numeric_data
        else:
            raise ValueError("No columns selected for embedding.")

        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(combined_embeddings)

        # Check for negative values
        min_distance = distance_matrix.min()
        st.write(f"Minimum distance value: {min_distance}")

        if np.any(distance_matrix < 0):
            st.write("Negative values found in distance matrix.")
            # Optionally, print the negative values
            st.write(distance_matrix[distance_matrix < 0])
        else:
            st.write("No negative values in distance matrix.")

        # Clip negative distances to zero
        distance_matrix[distance_matrix < 0] = 0
        
        # Flatten the similarity matrix
        num_records = len(data)
        indices = np.triu_indices(num_records, k=1)
        distances = distance_matrix[indices]
        similarities = (1 - (distances / distances.max())) * 100  # Normalize to 0-100%

        # Get record identifiers
        bdv_values = data[bdv_column].astype(str).values
        if 'id' in data.columns:
            ids = data['id'].values
        else:
            ids = data.index.values

        record1_indices = indices[0]
        record2_indices = indices[1]

        # Create DataFrame with results
        result = pd.DataFrame({
            'Record 1': bdv_values[record1_indices] + " (" + ids[record1_indices].astype(str) + ")",
            'Record 2': bdv_values[record2_indices] + " (" + ids[record2_indices].astype(str) + ")",
            'Similarity (%)': similarities
        })

        # Clustering
        if self.clustering_algorithm != 'None':
            if self.clustering_algorithm == 'KMeans':
                clustering = KMeans(n_clusters=self.num_clusters, random_state=42)
                labels = clustering.fit_predict(combined_embeddings)
            elif self.clustering_algorithm == 'AgglomerativeClustering':
                clustering = AgglomerativeClustering(n_clusters=self.num_clusters, affinity='euclidean', linkage='ward')
                labels = clustering.fit_predict(combined_embeddings)
            elif self.clustering_algorithm == 'DBSCAN':
                clustering = DBSCAN(metric='precomputed', eps=0.5, min_samples=2)
                labels = clustering.fit_predict(distance_matrix)
            elif self.clustering_algorithm == 'OPTICS':
                clustering = OPTICS(metric='precomputed', min_samples=2)
                labels = clustering.fit_predict(distance_matrix)
            else:
                raise ValueError(f"Unsupported clustering algorithm: {self.clustering_algorithm}")
            data['Cluster'] = labels
            # Include cluster labels in the result if desired
        else:
            data['Cluster'] = -1  # No clustering

        # Sort by Similarity (%) in decreasing order
        result = result.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

        # Store embeddings and labels for visualization
        self.combined_embeddings = combined_embeddings
        self.labels = data['Cluster'].values

        return result