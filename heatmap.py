import streamlit as st
import streamlit.components.v1 as components

# Replace with your actual GTM ID
GTM_ID = "GTM-P8FDLQQB"

# GTM script (head)
GTM_JS = f"""
<script>
(function(w,d,s,l,i){{w[l]=w[l]||[];w[l].push({{"gtm.start":
new Date().getTime(),event:"gtm.js"}});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
}})(window,document,'script','dataLayer','{GTM_ID}');
</script>
"""

# GTM noscript (body)
GTM_NOSCRIPT = f"""
<noscript>
<iframe src="https://www.googletagmanager.com/ns.html?id={GTM_ID}"
height="0" width="0" style="display:none;visibility:hidden"></iframe>
</noscript>
"""

# Inject GTM script
components.html(GTM_JS, height=0)
components.html(GTM_NOSCRIPT, height=0)

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

# Attempt to import GPU acceleration library
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class MatHeatmap:
    def __init__(self):
        """Initialize the cutting-edge gene expression heatmap app."""
        st.title("MatHeat: Gene Expression Heatmap Generator")
        st.write(
            "Upload your gene expression data file (CSV, TSV, XLSX, H5/HDF5, or JSON) and customize advanced parameters."
        )

    def load_data(self, file_obj, filename):
        """Load gene expression data from various file formats."""
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
        """Apply log transformation."""
        return np.log1p(data)

    def impute_missing_values(self, data, n_neighbors=5):
        """Impute missing values using k-NN imputation."""
        try:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed = imputer.fit_transform(data)
            return pd.DataFrame(imputed, index=data.index, columns=data.columns)
        except Exception as e:
            st.error(f"Error during imputation: {e}")
            return data

    def normalize_data(self, data, method='zscore'):
        """Normalize data using the selected method."""
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
        """Apply preprocessing steps: log transform, imputation, and normalization."""
        if apply_log:
            data = self.log_transform(data)
        data = self.impute_missing_values(data, n_neighbors=imputation_neighbors)
        data = self.normalize_data(data, method=normalization_method)
        return data

    def cluster_data(self, data, clustering_method, n_clusters=5):
        """
        Perform clustering using either KMeans or UMAP (with subsequent KMeans).
        For UMAP, both the clusters and the 2D embedding are returned.
        """
        if clustering_method == "KMeans":
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(data)
                return clusters
            except Exception as e:
                st.error(f"Error during KMeans clustering: {e}")
                return None
        elif clustering_method == "UMAP":
            try:
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding = reducer.fit_transform(data)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embedding)
                return clusters, embedding
            except Exception as e:
                st.error(f"Error during UMAP clustering: {e}")
                return None
        else:
            st.error("Unsupported clustering method.")
            return None

    def perform_enrichment_analysis(self, gene_list):
        """
        Perform Reactome pathway analysis using g:Profiler.
        """
        try:
            from gprofiler import GProfiler
            gp = GProfiler(return_dataframe=True)

            # Ensure gene_list is valid
            if not gene_list:
               st.error("No genes selected for enrichment analysis.")
               return None

            results = gp.profile(organism='hsapiens', query=gene_list, sources=['REAC'])

            if results.empty:
               st.warning("No significant Reactome pathways found.")
               return None
            results["Genes Queried"] = ', '.join(gene_list)

            return results
            
        except ImportError as e:
            st.error(f"g:Profiler import error: {e}\nPlease install it using 'pip install gprofiler-official'.")
            return None
        except Exception as e:
            st.error(f"Error during Reactome pathway analysis: {e}")
            return None

    def edge_detection(self, data):
        """
        Dummy edge detection using a simple gradient calculation.
        In a real-world scenario, you might apply advanced techniques (e.g., OpenCV).
        """
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

    def generate_heatmap(self, data, clusters=None):
        """Generate an interactive heatmap with adaptive color scaling."""
        try:
            flat_vals = data.values.flatten()
            vmin, vmax = np.percentile(flat_vals, 5), np.percentile(flat_vals, 95)
            fig = px.imshow(
                data,
                labels={"x": "Samples", "y": "Genes", "color": "Expression"},
                x=data.columns,
                y=data.index,
                zmin=vmin,
                zmax=vmax,
                color_continuous_scale='RdBu_r',
                title="Cutting-Edge Gene Expression Heatmap"
            )
            if clusters is not None:
                # Add cluster information in the hover text
                fig.update_traces(customdata=clusters, hovertemplate='Cluster: %{customdata}<extra></extra>')
            return fig
        except Exception as e:
            st.error(f"Error generating heatmap: {e}")
            return None

    def export_heatmap(self, fig, export_format="HTML"):
        """Provide export options for the heatmap (HTML/PNG)."""
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
        # Add a navigation section in the sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Home", "My Information", "Help", "Contact Us"])

        if page == "Home":
            # Existing Home page content (upload & settings, data processing, visualization, etc.)
            st.sidebar.header("Upload & Settings")
            uploaded_file = st.sidebar.file_uploader("Upload Gene Expression Data", type=["csv", "tsv", "xlsx", "h5", "hdf5", "json"])
            apply_log = st.sidebar.checkbox("Apply Log Transformation", value=True)
            normalization_method = st.sidebar.selectbox("Normalization Method", ["zscore", "minmax", "quantile"])
            imputation_neighbors = st.sidebar.slider("Imputation: Number of Neighbors", min_value=1, max_value=10, value=5)
            clustering_method = st.sidebar.selectbox("Clustering Method", ["None", "KMeans", "UMAP"])
            n_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=20, value=5)
            perform_enrichment = st.sidebar.checkbox("Perform Reactome Pathway Analysis", value=False)
            perform_edge_detection = st.sidebar.checkbox("Perform Edge Detection", value=False)
            export_format = st.sidebar.selectbox("Export Format", ["HTML", "PNG"])

            if uploaded_file is not None:
                data = self.load_data(uploaded_file, uploaded_file.name)
                if data is None:
                   return

                st.subheader("Data Preview")
                st.dataframe(data.head())

                # Preprocess data
                data_processed = self.preprocess_data(data, apply_log, normalization_method, imputation_neighbors)

                # Clustering (if selected)
                clusters = None
                if clustering_method != "None":
                    if clustering_method == "UMAP":
                        result = self.cluster_data(data_processed, clustering_method, n_clusters)
                        if result is not None:
                           clusters, embedding = result
                    else:
                        clusters = self.cluster_data(data_processed, clustering_method, n_clusters)

                # Reactome pathway analysis (if selected)
                enrichment_results = None
                if perform_enrichment:
                   gene_list = list(data_processed.index)
                   enrichment_results = self.perform_enrichment_analysis(gene_list)

                # Edge detection (if selected)
                edge_info = None
                if perform_edge_detection:
                   edge_info = self.edge_detection(data_processed)

                # Generate and display heatmap
                fig = self.generate_heatmap(data_processed, clusters)
                if fig is not None:
                   st.plotly_chart(fig)
                   self.export_heatmap(fig, export_format)

                # Display Reactome pathway analysis results in an expander
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

                # Display edge detection summary in an expander
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
            3. **Clustering:** Select a clustering method (KMeans or UMAP) and configure the number of clusters.
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
