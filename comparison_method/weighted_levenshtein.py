from comparison_method.base import ComparisonMethodBase
import pandas as pd
import numpy as np
import Levenshtein

class WeightedLevenshteinComparison(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "Weighted Levenshtein Distance"

    @staticmethod
    def get_parameter_info(data):
        # Generate dynamic parameters based on data columns
        columns = list(data.columns)
        default_weights = {col: 1.0 for col in columns if data[col].dtype == 'object'}
        parameter_info = [
            {
                'name': 'weights',
                'type': 'weights',
                'default': default_weights,
                'columns': list(default_weights.keys()),
                'description': 'Weights for each column (0 to 1)'
            }
        ]
        return parameter_info

    def __init__(self, **params):
        self.weights = params.get('weights', {})
        self.params = params

    def compare(self, data, bdv_column):
        # Check if bdv_column exists
        if bdv_column not in data.columns:
            raise ValueError(f"Column '{bdv_column}' not found in data.")

        # Prepare data
        bdv_values = data[bdv_column].astype(str).values

        if 'id' in data.columns:
            ids = data['id'].values
        else:
            ids = data.index.values

        num_records = len(data)
        # Create indices for all unique pairs
        indices = np.triu_indices(num_records, k=1)
        record1_indices = indices[0]
        record2_indices = indices[1]

        similarities = []
        for idx1, idx2 in zip(record1_indices, record2_indices):
            total_weight = 0.0
            similarity_sum = 0.0
            for col, weight in self.weights.items():
                val1 = str(data.iloc[idx1][col])
                val2 = str(data.iloc[idx2][col])
                max_len = max(len(val1), len(val2))
                if max_len == 0:
                    sim = 1.0
                else:
                    distance = Levenshtein.distance(val1, val2)
                    sim = 1 - distance / max_len
                similarity_sum += sim * weight
                total_weight += weight
            if total_weight > 0:
                overall_similarity = (similarity_sum / total_weight) * 100
            else:
                overall_similarity = 0.0
            similarities.append(overall_similarity)

        # Create DataFrame with results
        result = pd.DataFrame({
            'Record 1': bdv_values[record1_indices] + " (" + ids[record1_indices].astype(str) + ")",
            'Record 2': bdv_values[record2_indices] + " (" + ids[record2_indices].astype(str) + ")",
            'Similarity (%)': similarities
        })

        # Sort by Similarity (%) in decreasing order
        result = result.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

        return result