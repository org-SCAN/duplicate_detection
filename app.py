import streamlit as st
import pandas as pd
import importlib
import inspect
import pkgutil
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
import tempfile
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def load_comparison_methods():
    comparison_methods = {}
    import comparison_method

    package = comparison_method

    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if modname != 'base':
            module = importlib.import_module(f"comparison_method.{modname}")
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, comparison_method.ComparisonMethodBase)
                    and obj != comparison_method.ComparisonMethodBase
                ):
                    comparison_methods[obj.get_name()] = obj
    return comparison_methods

def main():
    st.title("Duplicate Detection Application")

    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data)

        # Store data in session state
        st.session_state['data'] = data
    else:
        st.stop()

    # Ask user to select bdv column
    st.header("2. Select Best Descriptive Value (bdv) Column")
    bdv_column = st.selectbox("Select the column that best describes your data", data.columns)
    st.session_state['bdv_column'] = bdv_column

    st.header("3. Select Comparison Method")
    comparison_methods = load_comparison_methods()
    method_name = st.selectbox("Select a method", list(comparison_methods.keys()))
    method_class = comparison_methods[method_name]
    st.session_state['method_name'] = method_name

    # Get parameters for the selected method
    st.header("4. Set Method Parameters")
    parameter_info = method_class.get_parameter_info(data)
    params = {}
    if parameter_info:
        for param in parameter_info:
            param_name = param.get('name')
            param_default = param.get('default')
            param_type = param.get('type', str)
            param_description = param.get('description', '')
            if param_type == int:
                params[param_name] = st.number_input(f"{param_name} ({param_description})", value=param_default, format="%d")
            elif param_type == float:
                params[param_name] = st.number_input(f"{param_name} ({param_description})", value=param_default)
            elif param_type == bool:
                params[param_name] = st.checkbox(f"{param_name} ({param_description})", value=param_default)
            elif param_type == 'weights':
                # Handle dynamic weights input
                weights = {}
                st.write("Set weights for each column (values between 0 and 1):")
                for col in param.get('columns'):
                    default_weight = param_default.get(col, 1.0)
                    weight = st.slider(f"Weight for '{col}'", min_value=0.0, max_value=1.0, value=default_weight)
                    weights[col] = weight
                params[param_name] = weights
            elif param_type == 'multi_select':
                options = param.get('options', [])
                params[param_name] = st.multiselect(f"{param_name} ({param_description})", options=options, default=param_default)
            elif param_type == 'select':
                options = param.get('options', [])
                default_index = options.index(param_default) if param_default in options else 0
                params[param_name] = st.selectbox(f"{param_name} ({param_description})", options=options, index=default_index)
            else:
                params[param_name] = st.text_input(f"{param_name} ({param_description})", value=param_default)
    else:
        st.write("No parameters to set for this method.")

    # Store parameters in session state
    st.session_state['params'] = params

    st.header("5. Run Comparison")
    if st.button("Run"):
        st.write("Running comparison...")
        method_instance = method_class(**params)
        result = method_instance.compare(data, bdv_column)
        # Store the result in session state
        st.session_state['result'] = result
        # Store embeddings and labels if available
        if hasattr(method_instance, 'combined_embeddings'):
            st.session_state['embeddings'] = method_instance.combined_embeddings
        else:
            st.session_state['embeddings'] = None
        if hasattr(method_instance, 'labels'):
            st.session_state['labels'] = method_instance.labels
        else:
            st.session_state['labels'] = None

    # Check if result is in session state
    if 'result' in st.session_state:
        result = st.session_state['result']
        st.write("Results:")
        # Interactive Data Table
        gb = GridOptionsBuilder.from_dataframe(result)
        gb.configure_pagination()
        gb.configure_side_bar()
        grid_options = gb.build()
        AgGrid(result, gridOptions=grid_options, enable_enterprise_modules=True)

        st.header("Visualizations")

        # Similarity Distribution Histogram
        st.subheader("Similarity Distribution")
        fig_hist = px.histogram(result, x='Similarity (%)', nbins=50)
        st.plotly_chart(fig_hist)

        # Heatmap of Similarity Scores
        st.subheader("Similarity Heatmap")
        # Create pivot table for heatmap
        pivot_table = result.pivot(index='Record 1', columns='Record 2', values='Similarity (%)')
        # To ensure square matrix, combine Record 1 and Record 2
        all_records = list(set(result['Record 1']) | set(result['Record 2']))
        pivot_table = pivot_table.reindex(index=all_records, columns=all_records)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_table, ax=ax)
        st.pyplot(fig)

        # Network Graph
        st.subheader("Similarity Network Graph")
        threshold = st.slider("Similarity Threshold (%)", min_value=0.0, max_value=100.0, value=80.0)
        G = nx.Graph()
        for idx, row in result.iterrows():
            if row['Similarity (%)'] >= threshold:
                G.add_edge(row['Record 1'], row['Record 2'], weight=row['Similarity (%)'])
        if len(G.nodes) > 0:
            nt = Network('800px', '800px', notebook=True)
            nt.from_nx(G)
            # Save and read graph as HTML file (streamlit components cannot render pyvis network directly)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            nt.save_graph(tmp_file.name)
            HtmlFile = open(tmp_file.name, 'r', encoding='utf-8')
            st.components.v1.html(HtmlFile.read(), height=800)
            HtmlFile.close()
            os.unlink(tmp_file.name)
        else:
            st.write("No edges above the threshold.")

        # Vector Visualizations
        embeddings = st.session_state.get('embeddings', None)
        labels = st.session_state.get('labels', None)
        if embeddings is not None:
            st.subheader("Vector Visualizations")
            vis_method = st.selectbox("Select visualization method", ['PCA', 't-SNE', 'UMAP'])
            if vis_method == 'PCA':
                reducer = PCA(n_components=2)
            elif vis_method == 't-SNE':
                n_samples = embeddings.shape[0]
                # Set perplexity to a value less than n_samples
                max_perplexity = min(30, n_samples - 1)
                perplexity = st.slider("Select t-SNE perplexity", min_value=5, max_value=int(max_perplexity), value=min(30, int((n_samples - 1)/2)))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            elif vis_method == 'UMAP':
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                reducer = PCA(n_components=2)

            embeddings_2d = reducer.fit_transform(embeddings)
            df_vis = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'label': labels if labels is not None else 0,
                'bdv': data[bdv_column].astype(str)
            })
            fig_vis = px.scatter(df_vis, x='x', y='y', color='label', hover_data=['bdv'])
            st.plotly_chart(fig_vis)

        # Download Results
        st.subheader("Download Results")
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Results as CSV", data=csv, file_name='similarity_results.csv', mime='text/csv')

    else:
        st.write("Click the 'Run' button to perform the comparison.")

if __name__ == "__main__":
    main()