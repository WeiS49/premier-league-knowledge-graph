"""
RAG pipeline for Premier League Football Knowledge Graph.
Converts natural language questions to SPARQL queries using Claude API.
Adapted from TD6 lab session code (modified to use Claude instead of Ollama).
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

import rdflib
from rdflib import Graph
from dotenv import load_dotenv
import anthropic

load_dotenv()

# --- Config ---
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # main model
FALLBACK_MODEL = "claude-3-haiku-20240307"

# path to our knowledge graph file
KG_PATH = Path(__file__).resolve().parent.parent.parent / "kg_artifacts" / "expanded.nt"
KG_FALLBACK = Path(__file__).resolve().parent.parent.parent / "kg_artifacts" / "football_kg.ttl"

# questions for testing (from project spec)
TEST_QUESTIONS = [
    "Which teams play in the Premier League?",
    "Who is the manager of Arsenal?",
    "What stadium does Manchester United play at?",
    "Which players are from England?",
    "When was Liverpool founded?",
]

SPARQL_INSTRUCTIONS = """You are a SPARQL expert. Generate a valid SPARQL 1.1 SELECT query
that answers the user's question based on the provided RDF schema.
Return ONLY the SPARQL query inside a ```sparql code block.
Do not include any explanation or text outside the code block.
Make sure to use the correct prefixes and predicates from the schema."""


# ---- 1. Load Graph ----
def load_graph(path=None):
    """Load RDF graph from file. Tries .nt first then .ttl fallback."""
    if path is None:
        if KG_PATH.exists():
            path = KG_PATH
        elif KG_FALLBACK.exists():
            path = KG_FALLBACK
        else:
            print(f"ERROR: No KG file found at {KG_PATH} or {KG_FALLBACK}")
            sys.exit(1)

    path = Path(path)
    g = Graph()
    fmt = "nt" if path.suffix == ".nt" else "turtle"
    print(f"Loading graph from {path} (format={fmt})...")
    g.parse(str(path), format=fmt)
    print(f"  -> Loaded {len(g)} triples")
    return g


# ---- 2. Get Prefix Block ----
def get_prefix_block(g):
    """Collect PREFIX declarations from graph namespace manager."""
    prefixes = []
    for prefix, uri in g.namespace_manager.namespaces():
        if prefix:  # skip default namespace
            prefixes.append(f"PREFIX {prefix}: <{uri}>")
    return "\n".join(prefixes)


# ---- 3. List Distinct Predicates ----
def list_distinct_predicates(g, limit=80):
    """Get distinct predicates used in the graph via SPARQL."""
    q = f"""
    SELECT DISTINCT ?p WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    results = g.query(q)
    preds = [str(row[0]) for row in results]
    return preds


# ---- 4. List Distinct Classes ----
def list_distinct_classes(g, limit=40):
    """Get distinct rdf:type classes in the graph."""
    q = f"""
    SELECT DISTINCT ?c WHERE {{
        ?s a ?c .
    }} LIMIT {limit}
    """
    results = g.query(q)
    classes = [str(row[0]) for row in results]
    return classes


# ---- 5. Sample Triples ----
def sample_triples(g, limit=20):
    """Get a sample of triples to show the LLM what data looks like."""
    q = f"""
    SELECT ?s ?p ?o WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    results = g.query(q)
    triples = []
    for row in results:
        triples.append(f"  {row[0]}  {row[1]}  {row[2]}")
    return triples


# ---- 6. Build Schema Summary ----
def build_schema_summary(g):
    """Combine prefixes, predicates, classes and sample triples into a schema summary string.
    This gets passed to the LLM so it knows what's in the KG."""

    prefix_block = get_prefix_block(g)
    predicates = list_distinct_predicates(g)
    classes = list_distinct_classes(g)
    samples = sample_triples(g)

    summary = f"""=== RDF Knowledge Graph Schema ===

PREFIXES:
{prefix_block}

PREDICATES ({len(predicates)} found):
{chr(10).join('- ' + p for p in predicates)}

CLASSES ({len(classes)} found):
{chr(10).join('- ' + c for c in classes)}

SAMPLE TRIPLES:
{chr(10).join(samples)}

=== End Schema ==="""

    print(f"Schema summary built: {len(predicates)} predicates, {len(classes)} classes")
    return summary


# ---- 7. Ask LLM ----
def ask_llm(prompt):
    """Send prompt to Claude API and get response.
    Adapted from lab - originally used Ollama, now using Anthropic SDK."""

    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        # try fallback model if main one fails
        print(f"  Warning: {CLAUDE_MODEL} failed ({e}), trying fallback...")
        try:
            message = client.messages.create(
                model=FALLBACK_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e2:
            print(f"  ERROR: Both models failed. {e2}")
            return f"[LLM ERROR: {e2}]"


# ---- 8. Make SPARQL Prompt ----
def make_sparql_prompt(schema_summary, question):
    """Create the prompt that asks Claude to generate SPARQL from natural language."""

    prompt = f"""{SPARQL_INSTRUCTIONS}

{schema_summary}

User question: {question}

Generate the SPARQL query:"""

    return prompt


# ---- 9. Extract SPARQL from Text ----
def extract_sparql_from_text(text):
    """Extract SPARQL query from LLM response (looks for code block).
    Adapted from TD6 lab regex pattern."""

    # try ```sparql ... ``` first
    match = re.search(r"```sparql\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # try generic code block
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # last resort: if it starts with SELECT or PREFIX, just use the whole thing
    if text.strip().upper().startswith(("SELECT", "PREFIX")):
        return text.strip()

    print("  Warning: Could not extract SPARQL from LLM response")
    return text.strip()


# ---- 10. Generate SPARQL ----
def generate_sparql(question, schema_summary):
    """Full NL->SPARQL: build prompt, call LLM, extract query."""
    prompt = make_sparql_prompt(schema_summary, question)
    raw_response = ask_llm(prompt)
    sparql = extract_sparql_from_text(raw_response)
    print(f"  Generated SPARQL:\n  {sparql[:200]}...")
    return sparql


# ---- 11. Run SPARQL ----
def run_sparql(g, query):
    """Execute SPARQL query on the graph. Returns (variable_names, rows)."""
    results = g.query(query)
    var_names = [str(v) for v in results.vars] if results.vars else []
    rows = []
    for row in results:
        rows.append([str(val) if val is not None else "" for val in row])
    return var_names, rows


# ---- 12. Repair SPARQL ----
def repair_sparql(schema_summary, question, bad_query, error_msg):
    """Self-repair: send the broken query + error back to Claude to fix.
    This is basically the retry mechanism from the lab."""

    repair_prompt = f"""The following SPARQL query was generated to answer: "{question}"

```sparql
{bad_query}
```

But it produced this error:
{error_msg}

Here is the RDF schema for reference:
{schema_summary}

Please fix the SPARQL query. Return ONLY the corrected query in a ```sparql code block."""

    print("  Attempting SPARQL repair...")
    response = ask_llm(repair_prompt)
    fixed = extract_sparql_from_text(response)
    return fixed


# ---- 13. Full RAG Pipeline ----
def answer_with_sparql_generation(g, schema_summary, question, try_repair=True):
    """Complete RAG pipeline: generate SPARQL, execute, optionally repair.
    Returns a dict with all the info."""

    result = {
        "question": question,
        "query": None,
        "vars": [],
        "rows": [],
        "repaired": False,
        "error": None,
    }

    # Step 1: generate SPARQL
    query = generate_sparql(question, schema_summary)
    result["query"] = query

    # Step 2: try to execute
    try:
        vars_, rows = run_sparql(g, query)
        result["vars"] = vars_
        result["rows"] = rows
        return result
    except Exception as e:
        print(f"  SPARQL execution failed: {e}")

        # Step 3: try repair if enabled
        if try_repair:
            try:
                fixed_query = repair_sparql(schema_summary, question, query, str(e))
                result["query"] = fixed_query
                result["repaired"] = True

                vars_, rows = run_sparql(g, fixed_query)
                result["vars"] = vars_
                result["rows"] = rows
                return result
            except Exception as e2:
                result["error"] = f"Repair also failed: {e2}"
                print(f"  Repair failed too: {e2}")
                return result
        else:
            result["error"] = str(e)
            return result


# ---- 14. Baseline (No RAG) ----
def answer_no_rag(question):
    """Baseline: just ask Claude directly without any KG context.
    For comparison with RAG approach."""

    prompt = f"""Answer this question about Premier League football concisely:
{question}"""

    return ask_llm(prompt)


# ---- 15. Pretty Print ----
def pretty_print_result(result):
    """Print RAG results in a readable format."""

    print("\n" + "=" * 60)
    print(f"Question: {result['question']}")
    print("-" * 60)

    if result.get("query"):
        print(f"SPARQL Query:\n{result['query']}")

    if result.get("repaired"):
        print("  [!] Query was repaired after initial failure")

    if result.get("error"):
        print(f"Error: {result['error']}")
    elif result.get("rows"):
        print(f"\nResults ({len(result['rows'])} rows):")
        if result.get("vars"):
            print("  " + " | ".join(result["vars"]))
            print("  " + "-" * 40)
        for row in result["rows"][:25]:  # dont print too many
            print("  " + " | ".join(row))
    else:
        print("  No results returned.")

    print("=" * 60)


# ---- CLI Demo ----
def run_all_questions(g, schema_summary):
    """Run all test questions and print comparison table."""

    print("\n" + "=" * 70)
    print("RUNNING ALL TEST QUESTIONS")
    print("=" * 70)

    results_table = []

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n--- Question {i}/{len(TEST_QUESTIONS)} ---")
        print(f"Q: {q}")

        # baseline
        print("\n[Baseline - No RAG]")
        baseline = answer_no_rag(q)
        print(f"  {baseline[:200]}")

        # RAG
        print("\n[RAG - SPARQL Generation]")
        rag_result = answer_with_sparql_generation(g, schema_summary, q)
        pretty_print_result(rag_result)

        results_table.append({
            "question": q,
            "baseline": baseline[:100],
            "rag_rows": len(rag_result.get("rows", [])),
            "repaired": rag_result.get("repaired", False),
            "error": rag_result.get("error"),
        })

    # print summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'#':<3} {'Question':<45} {'RAG Rows':<10} {'Repaired':<10} {'Error':<10}")
    print("-" * 78)
    for i, r in enumerate(results_table, 1):
        err = "Yes" if r["error"] else "No"
        print(f"{i:<3} {r['question'][:44]:<45} {r['rag_rows']:<10} {str(r['repaired']):<10} {err:<10}")


def main():
    parser = argparse.ArgumentParser(description="Football KG RAG Pipeline")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 5 test questions automatically")
    parser.add_argument("--kg", type=str, default=None, help="Path to KG file")
    args = parser.parse_args()

    # check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment!")
        print("Create a .env file with: ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    # load graph
    g = load_graph(args.kg)
    schema_summary = build_schema_summary(g)

    if args.run_all:
        run_all_questions(g, schema_summary)
        return

    # interactive mode
    print("\n" + "=" * 50)
    print("Football Knowledge Graph - RAG Demo (Claude API)")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        print()
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not question:
            continue

        # baseline answer
        print("\n--- Baseline (No RAG) ---")
        baseline = answer_no_rag(question)
        print(baseline)

        # RAG answer
        print("\n--- RAG (SPARQL Generation) ---")
        result = answer_with_sparql_generation(g, schema_summary, question)
        pretty_print_result(result)

        if result.get("repaired"):
            print("  Note: Self-repair was used to fix the initial SPARQL query.")


if __name__ == "__main__":
    main()
