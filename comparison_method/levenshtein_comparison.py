from comparison_method.base import ComparisonMethodBase
import pandas as pd
import numpy as np
import Levenshtein

class LevenshteinComparison(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "Levenshtein Distance"

    @staticmethod
    def get_parameter_info(data):
        # No additional parameters needed
        return []

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

        # Calculate Levenshtein distance and similarity
        similarities = []
        for idx1, idx2 in zip(record1_indices, record2_indices):
            val1 = bdv_values[idx1]
            val2 = bdv_values[idx2]
            max_len = max(len(val1), len(val2))
            if max_len == 0:
                similarity = 100.0
            else:
                distance = Levenshtein.distance(val1, val2)
                similarity = (1 - distance / max_len) * 100
            similarities.append(similarity)

        # Create DataFrame with results
        result = pd.DataFrame({
            'Record 1': bdv_values[record1_indices] + " (" + ids[record1_indices].astype(str) + ")",
            'Record 2': bdv_values[record2_indices] + " (" + ids[record2_indices].astype(str) + ")",
            'Similarity (%)': similarities
        })

        # Sort by Similarity (%) in decreasing order
        result = result.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

        return result