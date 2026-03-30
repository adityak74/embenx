import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import os
from core import Collection

st.set_page_config(page_title="Embenx Explorer 🚀", layout="wide")

st.title("Embenx Explorer 🚀")
st.markdown("Visualize and interact with your vector collections.")

# --- Sidebar: Collection Selection ---
st.sidebar.header("Collections")
parquet_files = glob.glob("*.parquet")
collection_names = [f.replace(".parquet", "") for f in parquet_files]

if not collection_names:
    st.info("No collections found in the current directory. Create one using the CLI or API.")
    st.stop()

selected_col_name = st.sidebar.selectbox("Select a Collection", collection_names)

# --- Load Data ---
@st.cache_data
def load_collection_data(name):
    path = f"{name}.parquet"
    df = pd.read_parquet(path)
    # Assume the vector column is named 'vector' as per Collection.to_parquet
    vectors = np.stack(df['vector'].values)
    return df, vectors

df, vectors = load_collection_data(selected_col_name)

st.sidebar.write(f"**Size:** {len(df)} documents")
st.sidebar.write(f"**Dimension:** {vectors.shape[1]}")

# --- Dimensionality Reduction ---
st.header("Vector Visualization")
method = st.radio("Reduction Method", ["PCA", "t-SNE"], horizontal=True)
dims = st.radio("Dimensions", [2, 3], horizontal=True)

@st.cache_data
def reduce_dims(vectors, method, dims):
    if method == "PCA":
        reducer = PCA(n_components=dims)
    else:
        # t-SNE can be slow for large datasets
        reducer = TSNE(n_components=dims, random_state=42)
    
    return reducer.fit_transform(vectors)

reduced_vectors = reduce_dims(vectors, method, dims)

# Create plotting dataframe
plot_df = df.copy()
if dims == 2:
    plot_df['x'] = reduced_vectors[:, 0]
    plot_df['y'] = reduced_vectors[:, 1]
    fig = px.scatter(plot_df, x='x', y='y', hover_data=df.columns.drop('vector'), 
                     title=f"{method} {dims}D Visualization")
else:
    plot_df['x'] = reduced_vectors[:, 0]
    plot_df['y'] = reduced_vectors[:, 1]
    plot_df['z'] = reduced_vectors[:, 2]
    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', hover_data=df.columns.drop('vector'),
                        title=f"{method} {dims}D Visualization")

st.plotly_chart(fig, use_container_width=True)

# --- Data Table ---
st.header("Metadata Inspector")
st.dataframe(df.drop(columns=['vector']), use_container_width=True)

# --- Interactive Search ---
st.header("Interactive Search")
query_text = st.text_input("Enter a search query")
top_k = st.slider("Top K", 1, 20, 5)

if query_text:
    with st.spinner("Searching..."):
        # We need an embedder to search
        # Defaulting to ollama/nomic-embed-text for now
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
