import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, dendrogram


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📊 Hierarchical Clustering Analysis")

st.write(
    "Implementation and analysis of Agglomerative and Divisive "
    "Hierarchical Clustering using the Mall Customers dataset."
)

st.info(
    "This application demonstrates distance measures, linkage methods, "
    "dendrograms, silhouette analysis, PCA visualization, and comparison "
    "of Agglomerative and Divisive clustering."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Settings")


# ---------------------------------------------------------
# DATASET UPLOAD
# ---------------------------------------------------------

st.sidebar.subheader("1. Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Mall_Customers.csv",
    type=["csv"]
)


if uploaded_file is None:

    st.warning(
        "Please upload the Mall_Customers.csv dataset using the "
        "sidebar."
    )

    st.markdown("""
    ### Required columns

    Your dataset should contain:

    - `CustomerID`
    - `Gender`
    - `Age`
    - `Annual Income (k$)`
    - `Spending Score (1-100)`

    Upload the CSV file from the sidebar to start the analysis.
    """)

    st.stop()


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv(uploaded_file)

st.success("Dataset loaded successfully!")


# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

st.header("1. Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Number of Records",
        df.shape[0]
    )

with col2:
    st.metric(
        "Number of Attributes",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


st.subheader("Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

st.dataframe(
    dtype_df,
    use_container_width=True
)


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

st.header("2. Data Preprocessing")

missing_values = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

st.subheader("Missing Values")

st.dataframe(
    missing_df,
    use_container_width=True
)


# ---------------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# ---------------------------------------------------------
# REMOVE IRRELEVANT ATTRIBUTES
# ---------------------------------------------------------

columns_to_remove = []

if "CustomerID" in df.columns:
    columns_to_remove.append("CustomerID")

if "Gender" in df.columns:
    columns_to_remove.append("Gender")


if len(columns_to_remove) > 0:

    processed_df = df.drop(
        columns=columns_to_remove
    )

else:

    processed_df = df.copy()


st.subheader("After Removing Irrelevant Attributes")

st.dataframe(
    processed_df.head(10),
    use_container_width=True
)


# ---------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------

st.subheader("Select Numerical Features")

available_features = processed_df.select_dtypes(
    include=np.number
).columns.tolist()


default_features = []

preferred_features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for feature in preferred_features:

    if feature in available_features:
        default_features.append(feature)


selected_features = st.multiselect(
    "Select features for clustering",
    available_features,
    default=default_features
)


if len(selected_features) < 2:

    st.warning(
        "Please select at least 2 numerical features."
    )

    st.stop()


X = processed_df[selected_features].copy()


# ---------------------------------------------------------
# STANDARDIZATION
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


st.success(
    "Numerical features standardized using StandardScaler."
)


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------

st.sidebar.subheader("2. Clustering")

cluster_method = st.sidebar.selectbox(
    "Clustering Method",
    [
        "Agglomerative",
        "Divisive"
    ]
)


number_of_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=5
)


# ---------------------------------------------------------
# DISTANCE MEASURE
# ---------------------------------------------------------

st.sidebar.subheader("3. Distance Measure")

distance_measure = st.sidebar.selectbox(
    "Distance Measure",
    [
        "Euclidean",
        "Manhattan",
        "Cosine"
    ]
)


distance_mapping = {
    "Euclidean": "euclidean",
    "Manhattan": "manhattan",
    "Cosine": "cosine"
}

selected_metric = distance_mapping[
    distance_measure
]


# ---------------------------------------------------------
# LINKAGE METHOD
# ---------------------------------------------------------

st.sidebar.subheader("4. Linkage Method")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    [
        "Single",
        "Complete",
        "Average",
        "Ward"
    ]
)


linkage_mapping = {
    "Single": "single",
    "Complete": "complete",
    "Average": "average",
    "Ward": "ward"
}

selected_linkage = linkage_mapping[
    linkage_method
]


# ---------------------------------------------------------
# DISTANCE MATRIX
# ---------------------------------------------------------

st.header("3. Distance Matrix")

distance_matrix = pairwise_distances(
    X_scaled,
    metric=selected_metric
)


st.write(
    "Selected Distance:",
    distance_measure
)


st.dataframe(
    pd.DataFrame(
        distance_matrix[:10, :10]
    ).round(3),
    use_container_width=True
)


# ---------------------------------------------------------
# DISTANCE MATRIX VISUALIZATION
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

image = ax.imshow(
    distance_matrix,
    cmap="viridis"
)

fig.colorbar(
    image,
    ax=ax
)

ax.set_title(
    distance_measure + " Distance Matrix"
)

ax.set_xlabel("Data Points")
ax.set_ylabel("Data Points")

st.pyplot(fig)


# ---------------------------------------------------------
# DENDROGRAM
# ---------------------------------------------------------

st.header("4. Hierarchical Dendrogram")


if selected_linkage == "Ward":

    Z = linkage(
        X_scaled,
        method="ward"
    )

else:

    Z = linkage(
        X_scaled,
        method=selected_linkage,
        metric=selected_metric
    )


fig, ax = plt.subplots(
    figsize=(14, 6)
)

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    ax=ax
)

ax.set_title(
    linkage_method +
    " Linkage Dendrogram"
)

ax.set_xlabel(
    "Data Points / Clusters"
)

ax.set_ylabel(
    "Distance"
)

st.pyplot(fig)


# ---------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------

st.header("5. Clustering Results")


if cluster_method == "Agglomerative":

    if selected_linkage == "Ward":

        model = AgglomerativeClustering(
            n_clusters=number_of_clusters,
            linkage="ward"
        )

    else:

        model = AgglomerativeClustering(
            n_clusters=number_of_clusters,
            linkage=selected_linkage,
            metric=selected_metric
        )

    labels = model.fit_predict(
        X_scaled
    )


else:

    # -----------------------------------------------------
    # DIVISIVE CLUSTERING
    # -----------------------------------------------------

    def divisive_clustering(
        X,
        n_clusters
    ):

        X = np.asarray(X)

        clusters = [
            np.arange(X.shape[0])
        ]

        while len(clusters) < n_clusters:

            largest_index = 0

            for i in range(
                1,
                len(clusters)
            ):

                if len(
                    clusters[i]
                ) > len(
                    clusters[largest_index]
                ):

                    largest_index = i

            selected_cluster = clusters.pop(
                largest_index
            )

            kmeans = KMeans(
                n_clusters=2,
                random_state=42,
                n_init=10
            )

            split_labels = kmeans.fit_predict(
                X[selected_cluster]
            )

            cluster_1 = selected_cluster[
                split_labels == 0
            ]

            cluster_2 = selected_cluster[
                split_labels == 1
            ]

            clusters.append(
                cluster_1
            )

            clusters.append(
                cluster_2
            )

        labels = np.zeros(
            X.shape[0],
            dtype=int
        )

        for cluster_id in range(
            len(clusters)
        ):

            labels[
                clusters[cluster_id]
            ] = cluster_id

        return labels


    labels = divisive_clustering(
        X_scaled,
        number_of_clusters
    )


# ---------------------------------------------------------
# SILHOUETTE SCORE
# ---------------------------------------------------------

score = silhouette_score(
    X_scaled,
    labels
)


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Number of Clusters",
        number_of_clusters
    )

with col2:

    st.metric(
        "Silhouette Score",
        round(score, 4)
    )


# ---------------------------------------------------------
# ADD CLUSTER LABELS
# ---------------------------------------------------------

result_df = processed_df.copy()

result_df[
    "Cluster"
] = labels


st.subheader(
    "Cluster Assignments"
)

st.dataframe(
    result_df.head(20),
    use_container_width=True
)


# ---------------------------------------------------------
# CLUSTER COUNTS
# ---------------------------------------------------------

st.subheader(
    "Number of Records in Each Cluster"
)

cluster_counts = (
    result_df["Cluster"]
    .value_counts()
    .sort_index()
)


st.bar_chart(
    cluster_counts
)


# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------

st.header("6. PCA Cluster Visualization")


pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": labels
})


fig, ax = plt.subplots(
    figsize=(10, 6)
)

scatter = ax.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df["Cluster"],
    cmap="viridis",
    s=50
)

ax.set_title(
    cluster_method +
    " Hierarchical Clustering"
)

ax.set_xlabel(
    "Principal Component 1"
)

ax.set_ylabel(
    "Principal Component 2"
)

fig.colorbar(
    scatter,
    ax=ax,
    label="Cluster"
)

st.pyplot(fig)


# ---------------------------------------------------------
# CLUSTER INTERPRETATION
# ---------------------------------------------------------

st.header("7. Cluster Interpretation")

cluster_summary = result_df.groupby(
    "Cluster"
)[selected_features].mean()


st.dataframe(
    cluster_summary.round(2),
    use_container_width=True
)


# ---------------------------------------------------------
# SILHOUETTE ANALYSIS FOR DIFFERENT K
# ---------------------------------------------------------

st.header(
    "8. Number of Clusters Analysis"
)

k_values = range(
    2,
    min(10, len(X_scaled) - 1) + 1
)

scores = []


for k in k_values:

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    temp_labels = model.fit_predict(
        X_scaled
    )

    temp_score = silhouette_score(
        X_scaled,
        temp_labels
    )

    scores.append(
        temp_score
    )


fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    list(k_values),
    scores,
    marker="o"
)

ax.set_title(
    "Silhouette Score for Different Numbers of Clusters"
)

ax.set_xlabel(
    "Number of Clusters"
)

ax.set_ylabel(
    "Silhouette Score"
)

ax.set_xticks(
    list(k_values)
)

st.pyplot(fig)


best_k = list(k_values)[
    np.argmax(scores)
]

best_score = max(scores)


st.success(
    "Best number of clusters: "
    + str(best_k)
    + " | Silhouette Score: "
    + str(round(best_score, 4))
)


# ---------------------------------------------------------
# DISTANCE COMPARISON
# ---------------------------------------------------------

st.header(
    "9. Distance Measure Comparison"
)

distance_results = []


for name, metric in [
    ("Euclidean", "euclidean"),
    ("Manhattan", "manhattan"),
    ("Cosine", "cosine")
]:

    model = AgglomerativeClustering(
        n_clusters=number_of_clusters,
        linkage="average",
        metric=metric
    )

    temp_labels = model.fit_predict(
        X_scaled
    )

    temp_score = silhouette_score(
        X_scaled,
        temp_labels
    )

    distance_results.append({
        "Distance": name,
        "Silhouette Score": temp_score
    })


distance_df = pd.DataFrame(
    distance_results
)


st.dataframe(
    distance_df.round(4),
    use_container_width=True
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    distance_df["Distance"],
    distance_df["Silhouette Score"]
)

ax.set_title(
    "Distance Measure Comparison"
)

ax.set_xlabel(
    "Distance Measure"
)

ax.set_ylabel(
    "Silhouette Score"
)

st.pyplot(fig)


# ---------------------------------------------------------
# LINKAGE COMPARISON
# ---------------------------------------------------------

st.header(
    "10. Linkage Method Comparison"
)

linkage_results = []


for name, method in [
    ("Single", "single"),
    ("Complete", "complete"),
    ("Average", "average"),
    ("Ward", "ward")
]:

    if method == "ward":

        model = AgglomerativeClustering(
            n_clusters=number_of_clusters,
            linkage="ward"
        )

    else:

        model = AgglomerativeClustering(
            n_clusters=number_of_clusters,
            linkage=method,
            metric="euclidean"
        )

    temp_labels = model.fit_predict(
        X_scaled
    )

    temp_score = silhouette_score(
        X_scaled,
        temp_labels
    )

    linkage_results.append({
        "Linkage": name,
        "Silhouette Score": temp_score
    })


linkage_df = pd.DataFrame(
    linkage_results
)


st.dataframe(
    linkage_df.round(4),
    use_container_width=True
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    linkage_df["Linkage"],
    linkage_df["Silhouette Score"]
)

ax.set_title(
    "Linkage Method Comparison"
)

ax.set_xlabel(
    "Linkage Method"
)

ax.set_ylabel(
    "Silhouette Score"
)

st.pyplot(fig)


# ---------------------------------------------------------
# AGGLOMERATIVE VS DIVISIVE
# ---------------------------------------------------------

st.header(
    "11. Agglomerative vs Divisive Comparison"
)


# Agglomerative
agg_model = AgglomerativeClustering(
    n_clusters=number_of_clusters,
    linkage="ward"
)

agg_labels = agg_model.fit_predict(
    X_scaled
)

agg_score = silhouette_score(
    X_scaled,
    agg_labels
)


# Divisive
div_labels = divisive_clustering(
    X_scaled,
    number_of_clusters
)

div_score = silhouette_score(
    X_scaled,
    div_labels
)


comparison_df = pd.DataFrame({
    "Method": [
        "Agglomerative",
        "Divisive"
    ],
    "Silhouette Score": [
        agg_score,
        div_score
    ]
})


st.dataframe(
    comparison_df.round(4),
    use_container_width=True
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    comparison_df["Method"],
    comparison_df["Silhouette Score"]
)

ax.set_title(
    "Agglomerative vs Divisive"
)

ax.set_xlabel(
    "Clustering Method"
)

ax.set_ylabel(
    "Silhouette Score"
)

st.pyplot(fig)


# ---------------------------------------------------------
# DOWNLOAD RESULTS
# ---------------------------------------------------------

st.header(
    "12. Download Results"
)

csv_data = result_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download Cluster Results",
    data=csv_data,
    file_name="Hierarchical_Clustering_Results.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# CONCLUSION
# ---------------------------------------------------------

st.header(
    "13. Conclusion"
)

st.write(
    "Hierarchical clustering was successfully applied to "
    "customer behaviour data. Different distance measures and "
    "linkage methods were compared using silhouette scores. "
    "Agglomerative clustering followed a bottom-up approach, "
    "while divisive clustering followed a top-down approach. "
    "PCA was used to visualize the resulting clusters."
)