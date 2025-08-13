
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import umap
from io import BytesIO
from umap import UMAP
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import IsolationForest
from gprofiler import GProfiler
import plotly.express as px
import plotly.graph_objects as go  
import plotly.io as pio
import plotly.express as px, plotly.io as pio
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

@st.cache_data
def fig_to_png(fig):
    return fig.to_image(format="png")

class MatHeatmap:
    def __init__(self):
        st.title("MatHeat: Gene Expression Heatmap Generator with ML")
        st.write("Upload your gene expression data file (CSV, TSV, XLSX, or JSON) and customize advanced parameters.")

    def load_data(self, file_obj, filename):
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(file_obj, index_col=0)
            elif filename.endswith('.tsv'):
                data = pd.read_csv(file_obj, sep='\t', index_col=0)
            elif filename.endswith('.xlsx'):
                data = pd.read_excel(file_obj, index_col=0)
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
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0.01)
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
            data_to_cluster = data
            cluster_labels = list(data.index)
        else:
            data_to_cluster = data.T
            cluster_labels = list(data.columns)
        if clustering_method == "KMeans":
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(data_to_cluster)
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
                cluster_dict = {cluster_labels[i]: clusters[i] for i in range(len(clusters))}
                return clusters, cluster_dict
            except Exception as e:
                st.error(f"Error during UMAP clustering: {e}")
                return None, None
        elif clustering_method == "Spectral":
            try:
                if data_to_cluster.shape[0] > 10000:
                    st.warning("Spectral Clustering may be slow for large datasets.")
                spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
                clusters = spectral.fit_predict(data_to_cluster)
                cluster_dict = {cluster_labels[i]: clusters[i] for i in range(len(clusters))}
                return clusters, cluster_dict
            except Exception as e:
                st.error(f"Error during Spectral clustering: {e}")
                return None, None
        else:
            return None, None

    def detect_anomalies(self, data, axis='genes', contamination=0.05):
        """Detect anomalies using Isolation Forest."""
        if axis == 'genes':
            data_to_analyze = data
            labels = data.index
        else:  
            data_to_analyze = data.T
            labels = data.columns
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(data_to_analyze)
        anomaly_indices = np.where(anomalies == -1)[0]
        return labels[anomaly_indices].tolist()

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

    def perform_de_analysis(self, data, group1_samples, group2_samples):
        try:
            de_results = []
            for gene in data.index:
                group1_values = data.loc[gene, group1_samples].values
                group2_values = data.loc[gene, group2_samples].values
                t_stat, p_val = stats.ttest_ind(group1_values, group2_values, equal_var=False)
                mean_diff = np.mean(group1_values) - np.mean(group2_values)
                de_results.append({'gene': gene, 'logFC': mean_diff, 't_stat': t_stat, 'p_val': p_val})
            de_df = pd.DataFrame(de_results)
            if not de_df.empty:
                de_df['adj_p_val'] = multipletests(de_df['p_val'], method='fdr_bh')[1]
            return de_df
        except Exception as e:
            st.error(f"Error during DE analysis: {e}")
            return None

    def generate_heatmap(self, data, clusters=None, cluster_dict=None, cluster_axis='samples', de_df=None, anomaly_labels=None, anomaly_axis='genes'):
        try:
            vmin, vmax = np.percentile(data.values.flatten(), 5), np.percentile(data.values.flatten(), 95)
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
            if cluster_dict:
                if cluster_axis == 'samples':
                    cluster_info = [[f"Cluster: {cluster_dict.get(sample, 'N/A')}" for sample in data.columns]] * len(data.index)
                else:
                    cluster_info = [[f"Cluster: {cluster_dict.get(gene, 'N/A')}"] * len(data.columns) for gene in data.index]
                fig.update_traces(
                    customdata=cluster_info,
                    hovertemplate=(
                        '<b>Sample</b>: %{x}<br>'
                        '<b>Gene</b>: %{y}<br>'
                        '<b>Expression</b>: %{z:.2f}<br>'
                        '%{customdata}<extra></extra>'
                    )
                )
            else:
                fig.update_traces(
                    hovertemplate=(
                        '<b>Sample</b>: %{x}<br>'
                        '<b>Gene</b>: %{y}<br>'
                        '<b>Expression</b>: %{z:.2f}<extra></extra>'
                    )
                )
            if de_df is not None:
                sig_genes = de_df[de_df['adj_p_val'] < 0.05]['gene'].tolist()
                for gene in sig_genes:
                    if gene in data.index:
                        row_idx = data.index.get_loc(gene)
                        fig.add_shape(
                            type="rect",
                            x0=-0.5,
                            y0=row_idx - 0.5,
                            x1=len(data.columns) - 0.5,
                            y1=row_idx + 0.5,
                            line=dict(color="yellow", width=2),
                            fillcolor="rgba(255,255,0,0.1)"
                        )
            if anomaly_labels:
                if anomaly_axis == 'genes':
                    for gene in anomaly_labels:
                        if gene in data.index:
                            row_idx = data.index.get_loc(gene)
                            fig.add_shape(
                                type="rect",
                                x0=-0.5,
                                y0=row_idx - 0.5,
                                x1=len(data.columns) - 0.5,
                                y1=row_idx + 0.5,
                                line=dict(color="red", width=2),
                                fillcolor="rgba(255,0,0,0.1)"
                            )
                else:  
                    for sample in anomaly_labels:
                        if sample in data.columns:
                            col_idx = data.columns.get_loc(sample)
                            fig.add_shape(
                                type="rect",
                                x0=col_idx - 0.5,
                                y0=-0.5,
                                x1=col_idx + 0.5,
                                y1=len(data.index) - 0.5,
                                line=dict(color="red", width=2),
                                fillcolor="rgba(255,0,0,0.1)"
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
                png_bytes = fig_to_png(fig)
                st.download_button("Download Heatmap as PNG", data=png_bytes, file_name="heatmap.png", mime="image/png")
            else:
                st.error("Unsupported export format.")
        except Exception as e:
            st.error(f"Error exporting heatmap: {e}")

    def run(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Home", "My Information", "Help", "Contact Us"])
        
        if page == "Home":
            st.sidebar.header("Upload & Settings")
            uploaded_file = st.sidebar.file_uploader("Upload Gene Expression Data", type=["csv", "tsv", "xlsx", "json"])
            apply_log = st.sidebar.checkbox("Apply Log Transformation", value=True)
            normalization_method = st.sidebar.selectbox("Normalization Method", ["zscore", "minmax", "quantile"])
            imputation_neighbors = st.sidebar.slider("Imputation: Number of Neighbors", min_value=1, max_value=10, value=5)
            clustering_method = st.sidebar.selectbox("Clustering Method", ["None", "KMeans", "UMAP", "Spectral"])
            cluster_axis = st.sidebar.selectbox("Cluster", ["samples", "genes"]) if clustering_method != "None" else None
            n_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=20, value=5)
            perform_enrichment = st.sidebar.checkbox("Perform Reactome Pathway Analysis", value=False)
            perform_edge_detection = st.sidebar.checkbox("Perform Edge Detection", value=False)
            export_format = st.sidebar.selectbox("Export Format", ["HTML", "PNG"])
            
            anomaly_detection = st.sidebar.checkbox("Detect Anomalies")
            anomaly_axis = st.sidebar.selectbox("Anomaly Axis", ["genes", "samples"]) if anomaly_detection else None
            contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.5, 0.05) if anomaly_detection else None
            
            if 'de_df' not in st.session_state:
                st.session_state.de_df = None
            if 'de_samples' not in st.session_state:
                st.session_state.de_samples = None
            if 'current_groups' not in st.session_state:
                st.session_state.current_groups = None
            
            if uploaded_file is not None:
                data = self.load_data(uploaded_file, uploaded_file.name)
                if data is None:
                    return
                st.subheader("Data Preview")
                st.write(f"Data shape: {data.shape} (rows = genes, columns = samples)")
                st.dataframe(data.head())
                data_processed = self.preprocess_data(data, apply_log, normalization_method, imputation_neighbors)
                
                with st.expander("View Processed Data Table"):
                    st.dataframe(data_processed)
                    csv_data = data_processed.to_csv().encode("utf-8")
                    st.download_button(
                        label="Download Processed Data as CSV",
                        data=csv_data,
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )
                
                clusters, cluster_dict = None, None
                if clustering_method != "None":
                    clusters, cluster_dict = self.cluster_data(
                        data_processed,
                        clustering_method,
                        n_clusters,
                        cluster_axis
                    )
                    if clusters is not None:
                        cluster_type = "Gene" if cluster_axis == "genes" else "Sample"
                        clusters_table = {}
                        for label, clus in cluster_dict.items():
                            clusters_table.setdefault(clus, []).append(label)
                        cluster_table_df = pd.DataFrame({
                            "Cluster": list(clusters_table.keys()),
                            f"{cluster_type}s": [", ".join(sorted(labels)) for labels in clusters_table.values()]
                        })
                        st.subheader(f"Cluster Assignments ({cluster_type}s)")
                        st.dataframe(cluster_table_df)
                
                anomaly_labels = None
                if anomaly_detection:
                    anomaly_labels = self.detect_anomalies(data_processed, anomaly_axis, contamination)
                    st.write(f"Anomalies in {anomaly_axis}:", anomaly_labels)
                
                st.sidebar.header("Differential Expression Analysis")
                group_method = st.sidebar.selectbox("Define Groups", ["From Clustering", "Manual Cluster Assignment", "From Uploaded File"])
                available_groups = []
                group_dict = None
                
                if group_method == "From Uploaded File":
                    group_file = st.sidebar.file_uploader("Upload Group Labels (CSV)", type=["csv"])
                    if group_file is not None:
                        group_df = pd.read_csv(group_file)
                        if len(group_df.columns) >= 2:
                            group_col = group_df.iloc[:, 0]
                            sample_col = group_df.iloc[:, 1]
                            group_dict = dict(zip(sample_col, group_col))
                            available_groups = list(set(group_col))
                        else:
                            st.error("Group file must have at least two columns")
                
                elif group_method == "Manual Cluster Assignment":
                    samples = list(data_processed.columns)
                    n_custom_clusters = st.sidebar.number_input(
                        "Number of Clusters", 
                        min_value=2, 
                        max_value=10, 
                        value=2,
                        key="custom_clusters"
                    )
                    manual_assignments = {}
                    st.sidebar.markdown("### Assign Samples to Clusters")
                    
                    for cluster_num in range(n_custom_clusters):
                        cluster_name = f"Cluster {cluster_num + 1}"
                        selected_samples = st.sidebar.multiselect(
                            f"{cluster_name} Samples",
                            options=samples,
                            key=f"cluster_{cluster_num}"
                        )
                        manual_assignments[cluster_name] = selected_samples
                    
                    assigned_samples = [s for samples in manual_assignments.values() for s in samples]
                    if len(assigned_samples) != len(samples):
                        st.sidebar.warning(f"Please assign all {len(samples)} samples to clusters")
                    else:
                        group_dict = {}
                        for cluster, samples_in_cluster in manual_assignments.items():
                            for sample in samples_in_cluster:
                                group_dict[sample] = cluster
                        available_groups = list(manual_assignments.keys())
                
                elif group_method == "From Clustering" and clustering_method != "None" and cluster_axis == 'samples':
                    available_groups = list(set(clusters))
                
                de_df = None
                if available_groups:
                    group1 = st.sidebar.selectbox("Select Group 1", available_groups)
                    group2 = st.sidebar.selectbox("Select Group 2", available_groups)
                    
                    st.session_state['current_groups'] = (group1, group2)
                    
                    if st.sidebar.button("Perform DE Analysis"):
                        if group_method == "From Uploaded File":
                            group1_samples = [sample for sample, grp in group_dict.items() if grp == group1]
                            group2_samples = [sample for sample, grp in group_dict.items() if grp == group2]
                        elif group_method == "Manual Cluster Assignment":
                            group1_samples = [sample for sample, grp in group_dict.items() if grp == group1]
                            group2_samples = [sample for sample, grp in group_dict.items() if grp == group2]
                        else:  # From Clustering
                            group1_samples = [sample for sample, clus in cluster_dict.items() if clus == group1]
                            group2_samples = [sample for sample, clus in cluster_dict.items() if clus == group2]
                        
                        st.session_state['de_samples'] = (group1_samples, group2_samples)
                        
                        de_df = self.perform_de_analysis(data_processed, group1_samples, group2_samples)
                        st.session_state['de_df'] = de_df

                de_df = st.session_state.get('de_df')
                if de_df is not None:
                    st.subheader("Differential Expression Results")
                    st.dataframe(de_df)
                    csv_data = de_df.to_csv().encode("utf-8")
                    st.download_button(
                        "Download DE Results as CSV",
                        data=csv_data,
                        file_name="de_results.csv",
                        mime="text/csv"
                    )

                    logfc_threshold = st.slider("LogFC Threshold", 0.0, 5.0, 1.0, key="volcano_logfc")
                    pval_threshold = st.slider("P-value Threshold", 0.001, 0.1, 0.05, key="volcano_pval")
                    show_gene_names = st.checkbox("Show gene names on plot", key="volcano_gene_names")
                    volcano_theme = st.selectbox("Volcano Plot Theme", 
                                               ['plotly', 'plotly_white', 'plotly_dark', 
                                                'ggplot2', 'seaborn', 'simple_white', 'none'], 
                                               key="volcano_theme")

                    de_df['Significance'] = np.where((abs(de_df['logFC']) > logfc_threshold) & 
                                                   (de_df['adj_p_val'] < pval_threshold),
                                                   'Significant', 'Not Significant')
                    fig_volcano = px.scatter(
                        de_df,
                        x='logFC',
                        y=-np.log10(de_df['adj_p_val']),
                        color='Significance',
                        hover_name='gene',
                        labels={'x': 'Log Fold Change', 'y': '-Log10 Adjusted P-value'},
                        title='Volcano Plot'
                    )
                    
                    if show_gene_names:
                        fig_volcano.add_trace(
                            go.Scatter(
                                x=de_df['logFC'],
                                y=-np.log10(de_df['adj_p_val']),
                                text=de_df['gene'],
                                mode='text',
                                name='Gene Names'
                            )
                        )
                    
                    if volcano_theme != 'none':
                        fig_volcano.update_layout(template=volcano_theme)
                    
                    fig_volcano.add_hline(y=-np.log10(pval_threshold), line_dash="dash")
                    fig_volcano.add_vline(x=logfc_threshold, line_dash="dash")
                    fig_volcano.add_vline(x=-logfc_threshold, line_dash="dash")
                    
                    st.plotly_chart(fig_volcano)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        volcano_html = fig_volcano.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            "Download Volcano Plot as HTML",
                            volcano_html,
                            "volcano_plot.html",
                            "text/html"
                        )
                    with col2:
                        volcano_png = pio.to_image(fig_volcano, format="png")
                        st.download_button(
                            "Download Volcano Plot as PNG",
                            volcano_png,
                            "volcano_plot.png",
                            "image/png"
                        )

                if st.sidebar.button("Clear DE Results"):
                    st.session_state.de_df = None
                    st.session_state.de_samples = None
                    st.session_state.current_groups = None
                    st.experimental_rerun()

                if clustering_method != "None" and cluster_axis == 'genes':
                    st.sidebar.subheader("Enrichment Analysis on Gene Clusters")
                    cluster_list = list(set(clusters))
                    selected_clusters = st.sidebar.multiselect("Select Clusters for Enrichment", cluster_list)
                    if selected_clusters:
                        selected_genes = [gene for gene, clus in cluster_dict.items() if clus in selected_clusters]
                        enrichment_results = self.perform_enrichment_analysis(selected_genes)
                        st.subheader("Enrichment Analysis on Selected Gene Clusters")
                        st.dataframe(enrichment_results)
                
                if de_df is not None:
                    sig_genes = de_df[de_df['adj_p_val'] < 0.05]['gene'].tolist()
                    if st.checkbox("Perform Enrichment on DE Genes"):
                        enrichment_results = self.perform_enrichment_analysis(sig_genes)
                        st.subheader("Enrichment Analysis on DE Genes")
                        st.dataframe(enrichment_results)
                
                enrichment_results = None
                if perform_enrichment:
                    gene_list = list(data_processed.index)
                    enrichment_results = self.perform_enrichment_analysis(gene_list)
                
                edge_info = None
                if perform_edge_detection:
                    edge_info = self.edge_detection(data_processed)
                
                fig = self.generate_heatmap(data_processed, clusters, cluster_dict, cluster_axis, de_df, anomaly_labels, anomaly_axis)
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
            st.write("Welcome to MatHeat! This application was developed by Seyyed Amin Seyyed Rezaei with a passion for bridging computational techniques and biological data analysis. Here you can perform DEGs analysis, generate advanced volcano plots and gene expression heatmaps, perform clustering, and run pathway analyses. Keywords: MatHeat, heatmap generator, gene expression, bioinformatics.")
            st.image("https://via.placeholder.com/300x200.png?text=Your+Photo ", caption="Your Name")
            st.write("For more about my work and background, please visit https://scholar.google.com/citations?user=pOLJKt4AAAAJ&hl=en. ")
        
        elif page == "Help":
            st.title("Help")
            st.write("**How to Use MatHeat: Gene Expression Heatmap Generator** 1. **Upload Data:** Use the sidebar to upload your gene expression data file in CSV, TSV, XLSX, or JSON format. 2. **Preprocessing Options:** Choose whether to apply a log transformation, select your normalization method, and set imputation parameters. 3. **Clustering:** Select a clustering method (KMeans, UMAP, or Spectral) and choose whether to cluster samples or genes. 4. **Anomaly Detection:** Optionally detect anomalies in genes or samples using Isolation Forest. 5. **Differential Expression:** Define groups via clustering or upload a group labels CSV, then perform DE analysis. 6. **Enrichment Analysis:** Analyze pathways for the whole dataset, gene clusters, or DE genes. 7. **Visualization & Export:** Generate an interactive heatmap and volcano plot, and export it as HTML or PNG.")
            st.write("If you encounter any issues or have questions, please refer to the documentation or contact us.")
        
        elif page == "Contact Us":
            st.title("Contact Us")
            st.write("If you have any questions, suggestions, or need support, please feel free to reach out: - **Email:** matheat.biology@gmail.com You can also follow us on social media: - [Twitter](https://twitter.com/matheat_biology ) - [LinkedIn](https://linkedin.com/in/yourprofile )")

if __name__ == '__main__':
    app = MatHeatmap()
    app.run()
