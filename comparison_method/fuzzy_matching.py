from comparison_method.base import ComparisonMethodBase
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import streamlit as st
from itertools import combinations

class FuzzyMatching(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "Fuzzy Matching"

    @staticmethod
    def get_parameter_info(data):
        columns = list(data.columns)
        parameter_info = []

        # Parameters for selecting columns
        parameter_info.append({
            'name': 'comparison_columns',
            'type': 'multi_select',
            'options': [col for col in columns if data[col].dtype == 'object'],
            'default': [],
            'description': 'Select text columns for fuzzy matching'
        })

        # Weights for each selected column
        default_weights = {col: 1.0 for col in columns}
        parameter_info.append({
            'name': 'weights',
            'type': 'weights',
            'default': default_weights,
            'columns': columns,
            'description': 'Weights for each column (0 to 1)'
        })

        parameter_info.append({
            'name': 'similarity_threshold',
            'type': float,
            'default': 80.0,
            'description': 'Similarity threshold (0 to 100)'
        })

        return parameter_info

    def __init__(self, **params):
        self.comparison_columns = params.get('comparison_columns', [])
        self.weights = params.get('weights', {})
        self.similarity_threshold = params.get('similarity_threshold', 80.0)
        self.params = params

    def compare(self, data, bdv_column):
        if not self.comparison_columns:
            raise ValueError("At least one column must be selected for comparison.")

        # Generate candidate pairs
        indices = list(data.index)
        candidate_pairs = list(combinations(indices, 2))

        bdv_values = data[bdv_column].astype(str).values
        if 'id' in data.columns:
            ids = data['id'].values
        else:
            ids = data.index.values

        record1_list = []
        record2_list = []
        similarities = []

        for idx1, idx2 in candidate_pairs:
            total_weight = 0.0
            similarity_sum = 0.0
            for col in self.comparison_columns:
                weight = self.weights.get(col, 1.0)
                val1 = str(data.loc[idx1, col])
                val2 = str(data.loc[idx2, col])

                # Compute fuzzy similarity
                sim = fuzz.token_set_ratio(val1, val2)
                similarity_sum += sim * weight
                total_weight += weight

            if total_weight > 0:
                overall_similarity = similarity_sum / total_weight
            else:
                overall_similarity = 0.0

            if overall_similarity >= self.similarity_threshold:
                record1 = f"{bdv_values[idx1]} ({ids[idx1]})"
                record2 = f"{bdv_values[idx2]} ({ids[idx2]})"
                record1_list.append(record1)
                record2_list.append(record2)
                similarities.append(overall_similarity)

        # Create DataFrame with results
        result = pd.DataFrame({
            'Record 1': record1_list,
            'Record 2': record2_list,
            'Similarity (%)': similarities
        })

        # Sort by Similarity (%) in decreasing order
        result = result.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

        return result