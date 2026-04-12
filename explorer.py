import glob
import json
import os

import litellm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from core import Collection

st.set_page_config(page_title="Embenx Explorer 🚀", layout="wide")

st.title("Embenx Explorer 🚀")
st.markdown("Visualize and interact with your vector collections.")

# --- Sidebar: Collection Selection & Management ---
st.sidebar.header("Collection Management")

parquet_files = glob.glob("*.parquet")
collection_names = [f.replace(".parquet", "") for f in parquet_files]

with st.sidebar.expander("➕ Create New Collection"):
    new_col_name = st.text_input("Collection Name")
    new_col_dim = st.number_input("Dimension", min_value=1, value=768)
    if st.button("Create"):
        if new_col_name:
            col = Collection(name=new_col_name, dimension=new_col_dim)
            # Create an empty file to represent the collection
            df = pd.DataFrame(columns=["vector"])
            df.to_parquet(f"{new_col_name}.parquet")
            st.success(f"Created {new_col_name}")
            st.rerun()

selected_col_name = st.sidebar.selectbox(
    "Select a Collection", collection_names if collection_names else ["None"]
)

if not selected_col_name or selected_col_name == "None":
    st.info("No collections found. Use the sidebar to create one or add data.")
    st.stop()


# --- Load Data ---
@st.cache_data
def load_collection_data(name):
    path = f"{name}.parquet"
    if not os.path.exists(path):
        return pd.DataFrame(), np.array([])
    df = pd.read_parquet(path)
    if df.empty or "vector" not in df.columns:
        return df, np.array([])
    vectors = np.stack(df["vector"].values)
    return df, vectors


df, vectors = load_collection_data(selected_col_name)

st.sidebar.write(f"**Size:** {len(df)} documents")
if vectors.size > 0:
    st.sidebar.write(f"**Dimension:** {vectors.shape[1]}")

tabs = st.tabs(
    [
        "Vector Clusters",
        "Metadata Inspector",
        "HNSW Visualizer 🕸️",
        "RAG Playground 💬",
        "Manage Data 🛠️",
        "Interactive Search",
    ]
)

# --- Tab 1: Vector Clusters ---
with tabs[0]:
    if vectors.size > 0:
        st.header("Vector Visualization")
        col1, col2 = st.columns(2)
        with col1:
            method = st.radio("Reduction Method", ["PCA", "t-SNE"], horizontal=True)
        with col2:
            dims = st.radio("Dimensions", [2, 3], horizontal=True)

        @st.cache_data
        def reduce_dims(vectors, method, dims):
            if len(vectors) < dims:
                return None

            if method == "PCA":
                reducer = PCA(n_components=dims)
            else:
                if len(vectors) <= dims:
                    return None
                reducer = TSNE(
                    n_components=dims, random_state=42, perplexity=min(30, len(vectors) - 1)
                )
            return reducer.fit_transform(vectors)

        reduced_vectors = reduce_dims(vectors, method, dims)

        if reduced_vectors is not None:
            plot_df = df.copy()
            if dims == 2:
                plot_df["x"] = reduced_vectors[:, 0]
                plot_df["y"] = reduced_vectors[:, 1]
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    hover_data=df.columns.drop("vector"),
                    title=f"{method} 2D Visualization",
                )
            else:
                plot_df["x"] = reduced_vectors[:, 0]
                plot_df["y"] = reduced_vectors[:, 1]
                plot_df["z"] = reduced_vectors[:, 2]
                fig = px.scatter_3d(
                    plot_df,
                    x="x",
                    y="y",
                    z="z",
                    hover_data=df.columns.drop("vector"),
                    title=f"{method} 3D Visualization",
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                f"Not enough data points to perform {method} {dims}D reduction. Need at least {dims + 1} samples."
            )
    else:
        st.info("Collection is empty. Add data in the 'Manage Data' tab.")

# --- Tab 2: Metadata Inspector ---
with tabs[1]:
    st.header("Metadata Inspector")
    if not df.empty:
        st.dataframe(df.drop(columns=["vector"], errors="ignore"), use_container_width=True)
    else:
        st.info("No metadata to display.")

# --- Tab 3: HNSW Visualizer ---
with tabs[2]:
    if vectors.size > 0:
        st.header("HNSW Graph Traversal (Virtual)")
        max_nodes = st.slider("Max Nodes to Visualize", 10, 500, min(100, len(vectors)))
        subset_indices = np.random.choice(len(vectors), min(len(vectors), max_nodes), replace=False)
        subset_vecs = vectors[subset_indices]

        if len(subset_vecs) >= 3:
            pca = PCA(n_components=3)
            coords_3d = pca.fit_transform(subset_vecs)

            layer_map = [
                np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]) for _ in range(len(subset_indices))
            ]

            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min(3, len(subset_vecs) - 1)).fit(coords_3d)
            distances, indices = nn.kneighbors(coords_3d)

            edge_x, edge_y, edge_z = [], [], []
            for i in range(len(subset_indices)):
                for neighbor_idx in indices[i]:
                    if i != neighbor_idx:
                        x0, y0, z0 = coords_3d[i]
                        x1, y1, z1 = coords_3d[neighbor_idx]
                        z0 += layer_map[i] * 5
                        z1 += layer_map[neighbor_idx] * 5
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_z.extend([z0, z1, None])

            edge_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                mode="lines",
            )
            node_x, node_y, node_z = [], [], []
            for i in range(len(subset_indices)):
                x, y, z = coords_3d[i]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z + layer_map[i] * 5)

            node_trace = go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers",
                marker=dict(showscale=True, colorscale="Viridis", color=layer_map, size=5),
            )
            fig_hnsw = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="HNSW Multi-Layer Navigation Graph",
                    scene=dict(
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False),
                        zaxis=dict(showticklabels=False),
                    ),
                ),
            )
            st.plotly_chart(fig_hnsw, use_container_width=True)
        else:
            st.warning("Need at least 3 points for 3D graph visualization.")
    else:
        st.info("Add data to see the HNSW graph.")

# --- Tab 4: RAG Playground ---
with tabs[3]:
    st.header("RAG Playground 💬")
    rag_model = st.text_input("LLM Model (LiteLLM format)", value="ollama/llama3")
    rag_query = st.text_area("Ask a question about this collection")

    if st.button("Generate RAG Response"):
        if not rag_query:
            st.warning("Please enter a question.")
        elif df.empty:
            st.error("Collection is empty.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                from llm import Embedder

                try:
                    emb = Embedder("ollama/nomic-embed-text")
                    q_vec = emb.embed_query(rag_query)
                    col = Collection.from_parquet(f"{selected_col_name}.parquet")
                    results = col.search(q_vec, top_k=3)
                    context = "\n".join(
                        [f"Context: {r[0].get('text', str(r[0]))}" for r in results]
                    )
                    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {rag_query}\n\nAnswer:"
                    response = litellm.completion(
                        model=rag_model, messages=[{"role": "user", "content": prompt}]
                    )
                    st.subheader("Answer")
                    st.write(response.choices[0].message.content)
                    with st.expander("Show Retrieved Context"):
                        st.text(context)
                except Exception as e:
                    st.error(f"RAG failed: {e}")

# --- Tab 5: Manage Data ---
with tabs[4]:
    st.header("Manage Data 🛠️")

    col_add1, col_add2 = st.columns(2)

    with col_add1:
        st.subheader("Add Single Document")
        manual_text = st.text_area("Document Text")
        manual_meta = st.text_input("Metadata (JSON)", value='{"category": "manual"}')
        if st.button("Add Document"):
            if manual_text:
                from llm import Embedder

                with st.spinner("Embedding..."):
                    emb = Embedder("ollama/nomic-embed-text")
                    vec = emb.embed_query(manual_text)
                    meta = json.loads(manual_meta)
                    meta["text"] = manual_text

                    col = Collection.from_parquet(f"{selected_col_name}.parquet")
                    col.add([vec], [meta])
                    col.to_parquet(f"{selected_col_name}.parquet")
                    st.success("Added document!")
                    st.rerun()

    with col_add2:
        st.subheader("Upload CSV/JSON")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "json"])
        text_col_name = st.text_input("Text Column Name", value="text")
        if uploaded_file is not None:
            if st.button("Process & Add"):
                if uploaded_file.name.endswith(".csv"):
                    up_df = pd.read_csv(uploaded_file)
                else:
                    up_df = pd.read_json(uploaded_file)

                if text_col_name not in up_df.columns:
                    st.error(f"Column '{text_col_name}' not found.")
                else:
                    from llm import Embedder

                    with st.spinner(f"Embedding {len(up_df)} rows..."):
                        emb = Embedder("ollama/nomic-embed-text")
                        texts = up_df[text_col_name].astype(str).tolist()
                        vectors = emb.embed_texts(texts)
                        metadata = up_df.to_dict(orient="records")

                        col = Collection.from_parquet(f"{selected_col_name}.parquet")
                        col.add(vectors, metadata)
                        col.to_parquet(f"{selected_col_name}.parquet")
                        st.success(f"Added {len(up_df)} items!")
                        st.rerun()

    st.divider()
    st.subheader("🔴 Danger Zone")
    if st.button("Delete This Collection"):
        os.remove(f"{selected_col_name}.parquet")
        st.warning(f"Deleted {selected_col_name}")
        st.rerun()

# --- Tab 6: Interactive Search ---
with tabs[5]:
    st.header("Interactive Search")
    query_text = st.text_input("Enter a search query", key="search_q")
    top_k = st.slider("Top K", 1, 20, 5, key="search_k")

    if query_text and not df.empty:
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
    elif df.empty:
        st.info("Collection is empty.")
