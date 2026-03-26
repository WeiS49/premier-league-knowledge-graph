"""
Streamlit web UI for the Football Knowledge Graph RAG demo.
Run with: streamlit run src/rag/app.py
"""

import json
import sys
from pathlib import Path

# ensure project root is on sys.path so imports work from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))

import streamlit as st
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# import our pipeline functions
from rag_pipeline import (
    load_graph,
    build_schema_summary,
    answer_no_rag,
    answer_with_sparql_generation,
)

# --- Page config ---
st.set_page_config(page_title="Football KG RAG", layout="wide")
st.title("Football Knowledge Graph - RAG Demo")
st.markdown("Compare baseline LLM answers vs RAG (SPARQL-based) answers using our Premier League knowledge graph.")


# --- Cache the graph loading so it doesn't reload every time ---
@st.cache_resource
def init_graph():
    """Load graph and build schema summary. Cached so we only do this once."""
    g = load_graph()
    schema = build_schema_summary(g)
    return g, schema


# --- Sidebar: KB statistics ---
st.sidebar.header("Knowledge Base Info")

stats_path = Path(__file__).resolve().parent.parent.parent / "kg_artifacts" / "stats.json"
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
    st.sidebar.metric("Triples", f"{stats.get('total_triples', 'N/A'):,}")
    st.sidebar.metric("Entities", f"{stats.get('unique_entities', 'N/A'):,}")
    st.sidebar.metric("Relations", f"{stats.get('unique_predicates', 'N/A'):,}")

    # show top predicates
    top_preds = stats.get("top_10_predicates", [])
    if top_preds:
        st.sidebar.markdown("**Top predicates:**")
        for p in top_preds[:5]:
            st.sidebar.text(f"  {p['short_name']}: {p['count']:,}")
else:
    st.sidebar.warning("stats.json not found in kg_artifacts/")
    st.sidebar.text("Run the KG build pipeline first.")

# load graph
try:
    g, schema_summary = init_graph()
    st.sidebar.success(f"Graph loaded: {len(g)} triples")
except Exception as e:
    st.error(f"Failed to load knowledge graph: {e}")
    st.stop()


# --- Main area ---
st.subheader("Ask a question about Premier League football")

# some example questions
examples = [
    "Which teams play in the Premier League?",
    "Who is the manager of Arsenal?",
    "What stadium does Manchester United play at?",
    "Which players are from England?",
    "When was Liverpool founded?",
]

# dropdown for examples
selected_example = st.selectbox("Or pick an example question:", [""] + examples)

question = st.text_input("Your question:", value=selected_example)

if st.button("Ask", type="primary") and question:
    col1, col2 = st.columns(2)

    # Left column: Baseline
    with col1:
        st.markdown("### Baseline (Direct LLM)")
        with st.spinner("Asking Claude directly..."):
            baseline_answer = answer_no_rag(question)
        st.write(baseline_answer)

    # Right column: RAG
    with col2:
        st.markdown("### RAG (Knowledge Graph)")
        with st.spinner("Generating SPARQL and querying KG..."):
            result = answer_with_sparql_generation(g, schema_summary, question)

        if result.get("error"):
            st.error(f"Error: {result['error']}")
        elif result.get("rows"):
            # show results as a table
            st.write(f"Found {len(result['rows'])} results:")

            # make a simple table
            if result["vars"] and result["rows"]:
                import pandas as pd
                df = pd.DataFrame(result["rows"], columns=result["vars"])
                st.dataframe(df, use_container_width=True)
            else:
                for row in result["rows"]:
                    st.text(" | ".join(row))
        else:
            st.info("No results found in the knowledge graph.")

        # show generated SPARQL in an expander
        with st.expander("View generated SPARQL query"):
            st.code(result.get("query", "N/A"), language="sparql")

        if result.get("repaired"):
            st.warning("Self-repair was used: the initial SPARQL query failed and was automatically corrected.")

elif not question:
    st.info("Enter a question above and click 'Ask' to get started!")
