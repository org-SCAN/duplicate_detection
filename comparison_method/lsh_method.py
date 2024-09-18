from comparison_method.base import ComparisonMethodBase
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
import streamlit as st

class LSHMethod(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "Locality-Sensitive Hashing (LSH)"

    @staticmethod
    def get_parameter_info(data):
        columns = list(data.columns)
        parameter_info = []

        # Parameters for selecting columns
        parameter_info.append({
            'name': 'text_column',
            'type': 'select',
            'options': [col for col in columns if data[col].dtype == 'object'],
            'default': columns[0],
            'description': 'Select the text column to use for LSH'
        })

        parameter_info.append({
            'name': 'threshold',
            'type': float,
            'default': 0.8,
            'description': 'Similarity threshold (0 to 1)'
        })

        parameter_info.append({
            'name': 'num_perm',
            'type': int,
            'default': 128,
            'description': 'Number of permutations for MinHash'
        })

        return parameter_info

    def __init__(self, **params):
        self.text_column = params.get('text_column')
        self.threshold = params.get('threshold', 0.8)
        self.num_perm = params.get('num_perm', 128)
        self.params = params

    def compare(self, data, bdv_column):
        if self.text_column not in data.columns:
            raise ValueError(f"Column '{self.text_column}' not found in data.")

        # Prepare MinHash signatures
        minhashes = {}
        for idx, row in data.iterrows():
            text = str(row[self.text_column])
            tokens = text.lower().split()
            m = MinHash(num_perm=self.num_perm)
            for token in tokens:
                m.update(token.encode('utf8'))
            minhashes[idx] = m

        # Build LSH index
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for idx, m in minhashes.items():
            lsh.insert(idx, m)

        # Query for similar items
        pairs = set()
        for idx, m in minhashes.items():
            result = lsh.query(m)
            for other_idx in result:
                if idx < other_idx:
                    pairs.add((idx, other_idx))

        # Prepare results
        bdv_values = data[bdv_column].astype(str).values
        if 'id' in data.columns:
            ids = data['id'].values
        else:
            ids = data.index.values

        similarities = []
        record1_list = []
        record2_list = []

        for idx1, idx2 in pairs:
            m1 = minhashes[idx1]
            m2 = minhashes[idx2]
            sim = m1.jaccard(m2) * 100  # Convert to percentage
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