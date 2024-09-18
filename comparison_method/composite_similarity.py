from comparison_method.base import ComparisonMethodBase
import pandas as pd
import numpy as np
import Levenshtein
from dateutil.parser import parse as parse_date

class CompositeSimilarity(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "Composite Similarity"

    @staticmethod
    def get_parameter_info(data):
        columns = list(data.columns)
        parameter_info = []

        # Parameters for selecting columns of each data type
        parameter_info.append({
            'name': 'string_columns',
            'type': 'multi_select',
            'options': [col for col in columns if data[col].dtype == 'object'],
            'default': [],
            'description': 'Select string columns for Levenshtein comparison'
        })

        parameter_info.append({
            'name': 'numeric_columns',
            'type': 'multi_select',
            'options': [col for col in columns if np.issubdtype(data[col].dtype, np.number)],
            'default': [],
            'description': 'Select numeric columns for numerical difference comparison'
        })

        parameter_info.append({
            'name': 'date_columns',
            'type': 'multi_select',
            'options': [col for col in columns if 'date' in col.lower()],
            'default': [],
            'description': 'Select date columns for date difference comparison'
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

        return parameter_info

    def __init__(self, **params):
        self.string_columns = params.get('string_columns', [])
        self.numeric_columns = params.get('numeric_columns', [])
        self.date_columns = params.get('date_columns', [])
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

            # String columns
            for col in self.string_columns:
                weight = self.weights.get(col, 1.0)
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

            # Numeric columns
            for col in self.numeric_columns:
                weight = self.weights.get(col, 1.0)
                val1 = data.iloc[idx1][col]
                val2 = data.iloc[idx2][col]
                max_val = data[col].max() - data[col].min() + 1e-5  # Avoid division by zero
                diff = abs(val1 - val2) / max_val
                sim = 1 - diff  # The smaller the difference, the higher the similarity
                similarity_sum += sim * weight
                total_weight += weight

            # Date columns
            for col in self.date_columns:
                weight = self.weights.get(col, 1.0)
                val1 = data.iloc[idx1][col]
                val2 = data.iloc[idx2][col]
                try:
                    date1 = parse_date(str(val1))
                    date2 = parse_date(str(val2))
                    date_range = data[col].apply(lambda x: parse_date(str(x))).max() - data[col].apply(lambda x: parse_date(str(x))).min()
                    max_days = date_range.days + 1e-5  # Avoid division by zero
                    diff = abs((date1 - date2).days) / max_days
                    sim = 1 - diff  # The smaller the difference, the higher the similarity
                except Exception:
                    sim = 0.0  # If parsing fails, consider similarity as 0
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