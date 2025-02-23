import streamlit as st
import streamlit.components.v1 as components

components.html("""
<head>
<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-P8FDLQQB');</script>
<!-- End Google Tag Manager -->
</head>
""", height=0)

import pandas as pd
import numpy as np
import json
import umap
from io import BytesIO
from umap import UMAP
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.cluster import KMeans
from gprofiler import GProfiler
import plotly.express as px
import plotly.io as pio

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class MatHeatmap:
    def __init__(self):
        st.title("MatHeat: Gene Expression Heatmap Generator")
        st.write(
            "Upload your gene expression data file (CSV, TSV, XLSX, H5/HDF5, or JSON) and customize advanced parameters."
        )

    def load_data(self, file_obj, filename):
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(file_obj, index_col=0)
            elif filename.endswith('.tsv'):
                data = pd.read_csv(file_obj, sep='\t', index_col=0)
            elif filename.endswith('.xlsx'):
                data = pd.read_excel(file_obj, index_col=0)
            elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                data = pd.read_hdf(file_obj)
            elif filename.endswith('.json'):
                data = pd.read_json(file_obj)
            else:
                st.error("Unsupported file format.")
                return None
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def log_transform(self, data):
        return np.log1p(data)

    def impute_missing_values(self, data, n_neighbors=5):
        try:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed = imputer.fit_transform(data)
            return pd.DataFrame(imputed, index=data.index, columns=data.columns)
        except Exception as e:
            st.error(f"Error during imputation: {e}")
            return data

    def normalize_data(self, data, method='zscore'):
        try:
            if method == 'zscore':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal')
            else:
                st.error("Unsupported normalization method.")
                return data
            scaled = scaler.fit_transform(data)
            return pd.DataFrame(scaled, index=data.index, columns=data.columns)
        except Exception as e:
            st.error(f"Error during normalization: {e}")
            return data

    def preprocess_data(self, data, apply_log, normalization_method, imputation_neighbors):
        if apply_log:
            data = self.log_transform(data)
        data = self.impute_missing_values(data, n_neighbors=imputation_neighbors)
        data = self.normalize_data(data, method=normalization_method)
        return data

    def cluster_data(self, data, clustering_method, n_clusters=5, cluster_axis='samples'):
        if cluster_axis == 'genes':
            data_to_cluster = data  # Cluster rows (genes)
            cluster_labels = list(data.index)
            st.write(f"Clustering genes: Data shape = {data_to_cluster.shape}, expecting {len(data.index)} clusters")
        else:  # samples
            data_to_cluster = data.T  # Transpose to cluster columns (samples)
            cluster_labels = list(data.columns)
            st.write(f"Clustering samples: Data shape = {data_to_cluster.shape}, expecting {len(data.columns)} clusters")

        if clustering_method == "KMeans":
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(data_to_cluster)
                st.write(f"KMeans clusters: {len(clusters)} assignments, {len(set(clusters))} unique clusters")
                cluster_dict = {cluster_labels[i]: clusters[i] for i in range(len(clusters))}
                return clusters, cluster_dict
            except Exception as e:
                st.error(f"Error during KMeans clustering: {e}")
                return None, None
        elif clustering_method == "UMAP":
            try:
                reducer = UMAP(n_components=2, random_state=42)
                embedding = reducer.fit_transform(data_to_cluster)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embedding)
                st.write(f"UMAP clusters: {len(clusters)} assignments, {len(set(clusters))} unique clusters")
                cluster_dict = {cluster_labels[i]: clusters[i] for i in range(len(clusters))}
                return clusters, cluster_dict
            except Exception as e:
                st.error(f"Error during UMAP clustering: {e}")
                return None, None
        else:
            st.error("Unsupported clustering method.")
            return None, None

    def perform_enrichment_analysis(self, gene_list):
        try:
            gp = GProfiler(return_dataframe=True)
            if not gene_list:
                st.error("No genes selected for enrichment analysis.")
                return None
            results = gp.profile(organism='hsapiens', query=gene_list, sources=['REAC'])
            if results.empty:
                st.warning("No significant Reactome pathways found.")
                return None
            results["Genes Queried"] = ', '.join(gene_list)
            return results
        except Exception as e:
            st.error(f"Error during Reactome pathway analysis: {e}")
            return None

    def edge_detection(self, data):
        try:
            grad_x = np.gradient(data.values, axis=0)
            grad_y = np.gradient(data.values, axis=1)
            edge_magnitude = np.abs(grad_x) + np.abs(grad_y)
            return {
                "Edge Mean": np.mean(edge_magnitude),
                "Edge Std": np.std(edge_magnitude)
            }
        except Exception as e:
            st.error(f"Error during edge detection: {e}")
            return None

    def generate_heatmap(self, data, clusters=None, cluster_dict=None, cluster_axis='samples'):
        try:
            vmin, vmax = np.percentile(data.values.flatten(), 5), np.percentile(data.values.flatten(), 95)
            st.write(f"Heatmap data shape: {data.shape}")

            fig = px.imshow(
                data,
                labels={"x": "Samples", "y": "Genes", "color": "Expression"},
                x=data.columns,
                y=data.index,
                zmin=vmin,
                zmax=vmax,
                color_continuous_scale='RdBu_r',
                title="Gene Expression Heatmap"
            )

            # If cluster_dict is provided, create custom hover data
            if cluster_dict:
                if cluster_axis == 'samples':
                    # Create a 2D array for samples
                    cluster_info = [[f"Cluster: {cluster_dict.get(sample, 'N/A')}" for sample in data.columns]] * len(data.index)
                else:  # genes
                    cluster_info = [[f"Cluster: {cluster_dict.get(gene, 'N/A')}"] * len(data.columns) for gene in data.index]

                fig.update_traces(
                    customdata=cluster_info,
                    hovertemplate=(
                        '<b>Sample</b>: %{x}<br>'
                        '<b>Gene</b>: %{y}<br>'
                        '<b>Expression</b>: %{z:.2f}<br>'
                        '%{customdata}<extra></extra>'
                    ),
                    hoverinfo='text'
                )
            else:
                st.write("No clusters provided")
                fig.update_traces(
                    hovertemplate=(
                        '<b>Sample</b>: %{x}<br>'
                        '<b>Gene</b>: %{y}<br>'
                        '<b>Expression</b>: %{z:.2f}<extra></extra>'
                    )
                )

            fig.update_layout(hovermode='closest')
            return fig

        except Exception as e:
            st.error(f"Error generating heatmap: {e}")
            return None

    def export_heatmap(self, fig, export_format="HTML"):
        try:
            if export_format == "HTML":
                html_bytes = fig.to_html(include_plotlyjs='cdn')
                st.download_button("Download Heatmap as HTML", data=html_bytes, file_name="heatmap.html", mime="text/html")
            elif export_format == "PNG":
                img_bytes = fig.to_image(format="png")
                st.download_button("Download Heatmap as PNG", data=img_bytes, file_name="heatmap.png", mime="image/png")
            else:
                st.error("Unsupported export format.")
        except Exception as e:
            st.error(f"Error exporting heatmap: {e}")

    def run(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Home", "My Information", "Help", "Contact Us"])

        if page == "Home":
            st.sidebar.header("Upload & Settings")
            uploaded_file = st.sidebar.file_uploader("Upload Gene Expression Data", type=["csv", "tsv", "xlsx", "h5", "hdf5", "json"])
            apply_log = st.sidebar.checkbox("Apply Log Transformation", value=True)
            normalization_method = st.sidebar.selectbox("Normalization Method", ["zscore", "minmax", "quantile"])
            imputation_neighbors = st.sidebar.slider("Imputation: Number of Neighbors", min_value=1, max_value=10, value=5)
            clustering_method = st.sidebar.selectbox("Clustering Method", ["None", "KMeans", "UMAP"])
            cluster_axis = st.sidebar.selectbox("Cluster", ["samples", "genes"]) if clustering_method != "None" else None
            n_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=20, value=5)
            perform_enrichment = st.sidebar.checkbox("Perform Reactome Pathway Analysis", value=False)
            perform_edge_detection = st.sidebar.checkbox("Perform Edge Detection", value=False)
            export_format = st.sidebar.selectbox("Export Format", ["HTML", "PNG"])

            if uploaded_file is not None:
                data = self.load_data(uploaded_file, uploaded_file.name)
                if data is None:
                    return

                st.subheader("Data Preview")
                st.write(f"Data shape: {data.shape} (rows = genes, columns = samples)")
                st.dataframe(data.head())

                data_processed = self.preprocess_data(data, apply_log, normalization_method, imputation_neighbors)

                clusters, cluster_dict = None, None
                if clustering_method != "None":
                    clusters, cluster_dict = self.cluster_data(
                        data_processed,
                        clustering_method,
                        n_clusters,
                        cluster_axis
                    )
                    if clusters is not None:
                        st.write(f"Clusters generated: {clusters}")
                    else:
                        st.error("Clustering failed, no clusters returned")

                enrichment_results = None
                if perform_enrichment:
                    gene_list = list(data_processed.index)
                    enrichment_results = self.perform_enrichment_analysis(gene_list)

                edge_info = None
                if perform_edge_detection:
                    edge_info = self.edge_detection(data_processed)

                fig = self.generate_heatmap(data_processed, clusters, cluster_dict, cluster_axis)
                if fig is not None:
                    st.plotly_chart(fig)
                    self.export_heatmap(fig, export_format)

                if perform_enrichment:
                    with st.expander("Reactome Pathway Analysis Results", expanded=True):
                        if isinstance(enrichment_results, pd.DataFrame):
                            st.dataframe(enrichment_results)
                            csv_data = enrichment_results.to_csv().encode("utf-8")
                            st.download_button(
                                "Download Pathway Analysis Results as CSV",
                                data=csv_data,
                                file_name="reactome_pathway_analysis.csv",
                                mime="text/csv"
                            )
                        else:
                            st.write(enrichment_results)

                if perform_edge_detection:
                    with st.expander("Edge Detection Summary", expanded=True):
                        st.write(edge_info)
                        json_data = json.dumps(edge_info, indent=4)
                        st.download_button(
                            "Download Edge Detection Summary as JSON",
                            data=json_data,
                            file_name="edge_detection_summary.json",
                            mime="application/json"
                        )

        elif page == "My Information":
            st.title("My Information")
            st.write("""
            Welcome to MatHeat! This application was developed by Seyyed Amin Seyyed Rezaei with a passion for 
            bridging computational techniques and biological data analysis. Here you can generate 
            advanced gene expression heatmaps, perform clustering, and run pathway analyses.
            Keywords: MatHeat, heatmap generator, gene expression, bioinformatics.
            """)
            st.image("https://via.placeholder.com/300x200.png?text=Your+Photo", caption="Your Name")
            st.write("For more about my work and background, please visit https://scholar.google.com/citations?user=pOLJKt4AAAAJ&hl=en.")

        elif page == "Help":
            st.title("Help")
            st.write("""
            **How to Use MatHeat: Gene Expression Heatmap Generator**
        
            1. **Upload Data:** Use the sidebar to upload your gene expression data file in CSV, TSV, XLSX, H5/HDF5, or JSON format.
            2. **Preprocessing Options:** Choose whether to apply a log transformation, select your normalization method, and set imputation parameters.
            3. **Clustering:** Select a clustering method (KMeans or UMAP) and choose whether to cluster samples or genes.
            4. **Additional Analyses:** Optionally perform Reactome pathway analysis or edge detection.
            5. **Visualization & Export:** Generate an interactive heatmap and export it as HTML or PNG.
            """)
            st.write("If you encounter any issues or have questions, please refer to the documentation or contact us.")

        elif page == "Contact Us":
            st.title("Contact Us")
            st.write("""
            If you have any questions, suggestions, or need support, please feel free to reach out:
        
            - **Email:** matheat.biology@gmail.com
        
            You can also follow us on social media:
            - [Twitter](https://twitter.com/yourhandle)
            - [LinkedIn](https://linkedin.com/in/yourprofile)
            """)

if __name__ == '__main__':
    app = MatHeatmap()
    app.run()
