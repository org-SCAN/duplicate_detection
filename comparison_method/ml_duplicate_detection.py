from comparison_method.base import ComparisonMethodBase
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import streamlit as st

class MLDuplicateDetection(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "Machine Learning Duplicate Detection"

    @staticmethod
    def get_parameter_info(data):
        columns = list(data.columns)
        parameter_info = []

        # Parameters for selecting columns
        parameter_info.append({
            'name': 'text_columns',
            'type': 'multi_select',
            'options': [col for col in columns if data[col].dtype == 'object'],
            'default': [],
            'description': 'Select text columns to use'
        })

        parameter_info.append({
            'name': 'numeric_columns',
            'type': 'multi_select',
            'options': [col for col in columns if np.issubdtype(data[col].dtype, np.number)],
            'default': [],
            'description': 'Select numeric columns to use'
        })

        parameter_info.append({
            'name': 'similarity_threshold',
            'type': float,
            'default': 0.5,
            'description': 'Similarity threshold (0 to 1)'
        })

        return parameter_info

    def __init__(self, **params):
        self.text_columns = params.get('text_columns', [])
        self.numeric_columns = params.get('numeric_columns', [])
        self.similarity_threshold = params.get('similarity_threshold', 0.5)
        self.params = params

    def compare(self, data, bdv_column):
        if not self.text_columns and not self.numeric_columns:
            raise ValueError("At least one column must be selected for comparison.")

        # Handle missing values
        for col in self.text_columns:
            data[col] = data[col].fillna('')

        # Generate candidate pairs
        indices = list(data.index)
        candidate_pairs = list(combinations(indices, 2))

        # Prepare features
        features = []
        for idx1, idx2 in candidate_pairs:
            row1 = data.loc[idx1]
            row2 = data.loc[idx2]
            feature_vector = []

            # Text column similarities
            for col in self.text_columns:
                text1 = str(row1[col])
                text2 = str(row2[col])
                similarity = self.compute_tfidf_similarity(text1, text2)
                feature_vector.append(similarity)

            # Numeric column differences
            for col in self.numeric_columns:
                val1 = row1[col]
                val2 = row2[col]
                diff = abs(val1 - val2)
                feature_vector.append(diff)

            features.append(feature_vector)

        # Check if features are empty
        if not features:
            raise ValueError("No features were generated. Check your data and selected columns.")

        # Convert features to DataFrame
        feature_names = [f"sim_{col}" for col in self.text_columns] + \
                        [f"diff_{col}" for col in self.numeric_columns]
        X = pd.DataFrame(features, columns=feature_names)

        # Synthetic labels (for demonstration)
        y = np.random.randint(0, 2, size=len(candidate_pairs))

        # Check if X is valid
        if X.isnull().values.any():
            X = X.fillna(0)

        # Train-test split
        X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(
            X, y, candidate_pairs, test_size=0.2, random_state=42
        )

        # Build and train the model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        model.fit(X_train, y_train)

        # Predict probabilities on test set
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compile results
        bdv_values = data[bdv_column].astype(str).values
        if 'id' in data.columns:
            ids = data['id'].values
        else:
            ids = data.index.values

        record1_list = []
        record2_list = []
        similarities = []

        for (idx, (idx1, idx2)) in enumerate(pairs_test):
            sim = y_proba[idx] * 100  # Convert to percentage
            if sim >= self.similarity_threshold * 100:
                record1 = f"{bdv_values[idx1]} ({ids[idx1]})"
                record2 = f"{bdv_values[idx2]} ({ids[idx2]})"
                record1_list.append(record1)
                record2_list.append(record2)
                similarities.append(sim)

        # Create DataFrame with results
        result = pd.DataFrame({
            'Record 1': record1_list,
            'Record 2': record2_list,
            'Similarity (%)': similarities
        })

        # Sort by Similarity (%) in decreasing order
        result = result.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

        return result

    def compute_tfidf_similarity(self, text1, text2):
        # Handle empty or missing texts
        documents = [text1 if text1 else '', text2 if text2 else '']
        vectorizer = TfidfVectorizer(stop_words=None)
        try:
            vectorizer.fit(documents)
            tfidf1 = vectorizer.transform([documents[0]])
            tfidf2 = vectorizer.transform([documents[1]])
            sim = (tfidf1 * tfidf2.T).toarray()[0][0]
        except ValueError:
            # If vocabulary is empty, return similarity of 0
            sim = 0.0
        return sim