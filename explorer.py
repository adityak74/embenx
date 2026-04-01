import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import os
import networkx as nx
from core import Collection

st.set_page_config(page_title="Embenx Explorer 🚀", layout="wide")

st.title("Embenx Explorer 🚀")
st.markdown("Visualize and interact with your vector collections.")

# --- Sidebar: Collection Selection ---
st.sidebar.header("Collections")
parquet_files = glob.glob("*.parquet")
collection_names = [f.replace(".parquet", "") for f in parquet_files]

if not collection_names:
    st.info("No collections found. Create one using the CLI or API.")
    st.stop()

selected_col_name = st.sidebar.selectbox("Select a Collection", collection_names)

# --- Load Data ---
@st.cache_data
def load_collection_data(name):
    path = f"{name}.parquet"
    df = pd.read_parquet(path)
    vectors = np.stack(df['vector'].values)
    return df, vectors

df, vectors = load_collection_data(selected_col_name)

st.sidebar.write(f"**Size:** {len(df)} documents")
st.sidebar.write(f"**Dimension:** {vectors.shape[1]}")

tabs = st.tabs(["Vector Clusters", "Metadata Inspector", "HNSW Visualizer 🕸️", "Interactive Search"])

# --- Tab 1: Vector Clusters ---
with tabs[0]:
    st.header("Vector Visualization")
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio("Reduction Method", ["PCA", "t-SNE"], horizontal=True)
    with col2:
        dims = st.radio("Dimensions", [2, 3], horizontal=True)

    @st.cache_data
    def reduce_dims(vectors, method, dims):
        if method == "PCA":
            reducer = PCA(n_components=dims)
        else:
            reducer = TSNE(n_components=dims, random_state=42)
        return reducer.fit_transform(vectors)

    reduced_vectors = reduce_dims(vectors, method, dims)

    plot_df = df.copy()
    if dims == 2:
        plot_df['x'] = reduced_vectors[:, 0]
        plot_df['y'] = reduced_vectors[:, 1]
        fig = px.scatter(plot_df, x='x', y='y', hover_data=df.columns.drop('vector'), 
                         title=f"{method} 2D Visualization")
    else:
        plot_df['x'] = reduced_vectors[:, 0]
        plot_df['y'] = reduced_vectors[:, 1]
        plot_df['z'] = reduced_vectors[:, 2]
        fig = px.scatter_3d(plot_df, x='x', y='y', z='z', hover_data=df.columns.drop('vector'),
                            title=f"{method} 3D Visualization")

    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Metadata Inspector ---
with tabs[1]:
    st.header("Metadata Inspector")
    st.dataframe(df.drop(columns=['vector']), use_container_width=True)

# --- Tab 3: HNSW Visualizer ---
with tabs[2]:
    st.header("HNSW Graph Traversal (Virtual)")
    st.markdown("""
    This view simulates the **Hierarchical Navigable Small World (HNSW)** structure.
    Nodes are assigned to layers (Level 0 being the most dense).
    """)
    
    # Take a subset for performance
    max_nodes = st.slider("Max Nodes to Visualize", 50, 500, 100)
    subset_indices = np.random.choice(len(vectors), min(len(vectors), max_nodes), replace=False)
    subset_vecs = vectors[subset_indices]
    
    # 3D PCA for node positions
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(subset_vecs)
    
    # Assign layers
    layers = 3
    layer_map = [] # list of (node_idx, layer_id)
    for i in range(len(subset_indices)):
        # Higher layers are exponentially rarer
        r = np.random.random()
        if r < 0.1: l = 2
        elif r < 0.3: l = 1
        else: l = 0
        layer_map.append(l)
    
    # Build Edges (KNN within and across layers)
    edges = []
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(coords_3d)
    distances, indices = nn.kneighbors(coords_3d)
    
    for i in range(len(subset_indices)):
        for neighbor_idx in indices[i]:
            if i != neighbor_idx:
                edges.append((i, neighbor_idx))

    # Plotly 3D Graph
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in edges:
        x0, y0, z0 = coords_3d[edge[0]]
        x1, y1, z1 = coords_3d[edge[1]]
        # Offset z by layer height
        z0 += layer_map[edge[0]] * 5
        z1 += layer_map[edge[1]] * 5
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=go.scatter3d.Line(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_z = []
    for i in range(len(subset_indices)):
        x, y, z = coords_3d[i]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z + layer_map[i] * 5)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=layer_map,
            size=5,
            colorbar=dict(title='HNSW Layer'),
            line_width=2
        )
    )

    fig_hnsw = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='HNSW Multi-Layer Navigation Graph',
                    showlegend=False,
                    scene=dict(
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ),
                    margin=dict(b=0, l=0, r=0, t=40)
                ))
    
    st.plotly_chart(fig_hnsw, use_container_width=True)

# --- Tab 4: Interactive Search ---
with tabs[3]:
    st.header("Interactive Search")
    query_text = st.text_input("Enter a search query")
    top_k = st.slider("Top K", 1, 20, 5)

    if query_text:
        with st.spinner("Searching..."):
            from llm import Embedder
            try:
                emb = Embedder("ollama/nomic-embed-text")
                q_vec = emb.embed_query(query_text)
                
                col = Collection.from_parquet(f"{selected_col_name}.parquet")
                results = col.search(q_vec, top_k=top_k)
                
                st.subheader("Search Results")
                for i, (meta, dist) in enumerate(results):
                    with st.expander(f"Result {i+1} (Score: {dist:.4f})"):
                        st.write(meta)
            except Exception as e:
                st.error(f"Search failed: {e}")
