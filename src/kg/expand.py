"""
Knowledge Base expansion using Wikidata SPARQL endpoint.
Takes aligned entities and pulls additional triples from Wikidata
to enrich the football knowledge graph.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from SPARQLWrapper import JSON, SPARQLWrapper

FB = Namespace("http://example.org/football/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KG_ARTIFACTS = PROJECT_ROOT / "kg_artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# predicates we care about for expansion
KEY_PREDICATES = ["P54", "P27", "P286", "P118", "P413", "P17", "P115", "P569", "P571"]

# predicates to filter out (too generic or not useful)
BORING_PREDICATES = [
    "schema.org", "wikiba.se", "www.w3.org/2004/02/skos",
    "www.w3.org/ns/prov", "www.wikidata.org/prop/statement",
    "www.wikidata.org/prop/qualifier", "www.wikidata.org/prop/reference",
    "www.wikidata.org/prop/P", "commons.wikimedia.org",
    "upload.wikimedia.org", "www.wikidata.org/value",
]

BATCH_SIZE = 20  # entities per VALUES batch query
RATE_LIMIT_DELAY = 2  # seconds between queries


def make_sparql():
    """Create a SPARQLWrapper instance with proper headers."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "PremierLeagueKG-StudentProject/1.0 (ESILV coursework)")
    return sparql


def run_query(sparql, query, retries=3):
    """Execute SPARQL query with retry logic."""
    for attempt in range(retries):
        try:
            sparql.setQuery(query)
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print(f"    Query failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                wait = (attempt + 1) * 5  # exponential-ish backoff
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Giving up on this query")
                return []


def is_boring(pred_uri):
    """Check if a predicate is one we want to filter out."""
    pred_str = str(pred_uri)
    for boring in BORING_PREDICATES:
        if boring in pred_str:
            return True
    return False


def parse_binding(binding_data):
    """Parse a SPARQL result binding into an rdflib term. Returns None for bnodes."""
    if binding_data["type"] == "uri":
        return URIRef(binding_data["value"])
    elif binding_data["type"] == "literal":
        if "datatype" in binding_data:
            return Literal(binding_data["value"], datatype=URIRef(binding_data["datatype"]))
        else:
            return Literal(binding_data["value"])
    return None  # skip bnodes


def make_values_clause(qids):
    """Build a SPARQL VALUES clause from a list of QIDs."""
    return " ".join(f"wd:{q}" for q in qids)


# --- Strategy 1: Batch 1-hop expansion ---

def expand_batch(sparql, qid_batch, expanded_g):
    """Get all wdt: triples for a batch of entities using VALUES."""
    values_str = make_values_clause(qid_batch)
    query = f"""
    SELECT ?s ?p ?o WHERE {{
        VALUES ?s {{ {values_str} }}
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
    }} LIMIT 10000
    """
    results = run_query(sparql, query)
    count = 0
    for r in results:
        pred = URIRef(r["p"]["value"])
        if is_boring(pred):
            continue
        obj = parse_binding(r["o"])
        if obj is None:
            continue
        subj = URIRef(r["s"]["value"])
        expanded_g.add((subj, pred, obj))
        count += 1
    return count


# --- Strategy 2: Reverse expansion ---

def expand_reverse(sparql, qid_batch, expanded_g):
    """Find entities that point TO our entities (e.g. players → teams)."""
    values_str = make_values_clause(qid_batch)
    query = f"""
    SELECT ?s ?p ?o WHERE {{
        VALUES ?o {{ {values_str} }}
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        FILTER(STRSTARTS(STR(?s), "http://www.wikidata.org/entity/Q"))
    }} LIMIT 10000
    """
    results = run_query(sparql, query)
    count = 0
    for r in results:
        pred = URIRef(r["p"]["value"])
        if is_boring(pred):
            continue
        subj = URIRef(r["s"]["value"])
        obj = URIRef(r["o"]["value"])
        expanded_g.add((subj, pred, obj))
        count += 1
    return count


# --- Strategy 3: Predicate-controlled batch ---

def expand_predicate_batch(sparql, team_qids, expanded_g):
    """For each key predicate, batch-query all teams to pull related entities.
    E.g. find all players who played for (P54) any of our teams."""
    total = 0
    values_str = make_values_clause(team_qids)

    # reverse direction: ?s wdt:Pxx ?team (players who played for team, coaches of team, etc.)
    reverse_preds = ["P54", "P286", "P118"]  # member of sports team, head coach, league
    for pid in reverse_preds:
        query = f"""
        SELECT ?s ?o WHERE {{
            VALUES ?o {{ {values_str} }}
            ?s wdt:{pid} ?o .
            FILTER(STRSTARTS(STR(?s), "http://www.wikidata.org/entity/Q"))
        }} LIMIT 20000
        """
        results = run_query(sparql, query)
        for r in results:
            subj = URIRef(r["s"]["value"])
            obj = URIRef(r["o"]["value"])
            expanded_g.add((subj, WDT[pid], obj))
            total += 1
        print(f"    wdt:{pid} reverse: +{len(results)} triples")
        time.sleep(RATE_LIMIT_DELAY)

    # forward direction: ?team wdt:Pxx ?o (team's country, league, stadium)
    forward_preds = ["P17", "P115", "P571", "P159"]  # country, stadium, inception, HQ location
    for pid in forward_preds:
        query = f"""
        SELECT ?s ?o WHERE {{
            VALUES ?s {{ {values_str} }}
            ?s wdt:{pid} ?o .
        }} LIMIT 5000
        """
        results = run_query(sparql, query)
        for r in results:
            subj = URIRef(r["s"]["value"])
            obj_data = r["o"]
            obj = parse_binding(obj_data)
            if obj is None:
                continue
            expanded_g.add((subj, WDT[pid], obj))
            total += 1
        print(f"    wdt:{pid} forward: +{len(results)} triples")
        time.sleep(RATE_LIMIT_DELAY)

    return total


# --- Strategy 4: Enriched 2-hop expansion ---

def expand_2hop_batch(sparql, qid_batch, expanded_g):
    """Get connected entities for a batch, then batch-query their properties too."""
    values_str = make_values_clause(qid_batch)

    # step 1: find connected entities
    query = f"""
    SELECT DISTINCT ?root ?connected WHERE {{
        VALUES ?root {{ {values_str} }}
        ?root ?p ?connected .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        FILTER(STRSTARTS(STR(?connected), "http://www.wikidata.org/entity/Q"))
    }} LIMIT 2000
    """
    results = run_query(sparql, query)
    connected_qids = set()
    for r in results:
        cqid = r["connected"]["value"].split("/")[-1]
        connected_qids.add(cqid)

    if not connected_qids:
        return 0

    print(f"    Found {len(connected_qids)} connected entities, fetching their triples...")

    # step 2: batch-query connected entities in chunks
    connected_list = list(connected_qids)
    count = 0
    chunk_size = 30  # smaller chunks for 2-hop to avoid timeouts
    for i in range(0, len(connected_list), chunk_size):
        chunk = connected_list[i:i+chunk_size]
        chunk_values = make_values_clause(chunk)
        query2 = f"""
        SELECT ?s ?p ?o WHERE {{
            VALUES ?s {{ {chunk_values} }}
            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT 5000
        """
        results2 = run_query(sparql, query2)
        for r2 in results2:
            pred = URIRef(r2["p"]["value"])
            if is_boring(pred):
                continue
            obj = parse_binding(r2["o"])
            if obj is None:
                continue
            subj = URIRef(r2["s"]["value"])
            expanded_g.add((subj, pred, obj))
            count += 1
        time.sleep(RATE_LIMIT_DELAY)

    return count


# --- Strategy 5: Enrich discovered entities ---

def enrich_new_entities(sparql, expanded_g, known_qids):
    """Take entities discovered via reverse queries and get their basic properties.
    E.g. for players found via P54 reverse, get their birth date, nationality, position."""
    # collect entity QIDs that appeared as subjects but weren't in original set
    new_qids = set()
    for s, p, o in expanded_g:
        s_str = str(s)
        if s_str.startswith("http://www.wikidata.org/entity/Q"):
            qid = s_str.split("/")[-1]
            if qid not in known_qids:
                new_qids.add(qid)

    if not new_qids:
        return 0

    MAX_ENRICH = 3000
    print(f"  Found {len(new_qids)} new entities to enrich (capping at {MAX_ENRICH})")

    # only enrich with key football predicates to keep it focused
    enrich_preds = ["P27", "P569", "P413", "P54", "P1532", "P21", "P19"]  # nationality, DOB, position, team, country for sport, sex, birthplace
    new_list = list(new_qids)[:MAX_ENRICH]
    total = 0

    for i in range(0, len(new_list), BATCH_SIZE):
        batch = new_list[i:i+BATCH_SIZE]
        values_str = make_values_clause(batch)
        preds_filter = " ".join(f"wdt:{p}" for p in enrich_preds)

        for pid in enrich_preds:
            query = f"""
            SELECT ?s ?o WHERE {{
                VALUES ?s {{ {values_str} }}
                ?s wdt:{pid} ?o .
            }} LIMIT 5000
            """
            results = run_query(sparql, query)
            for r in results:
                subj = URIRef(r["s"]["value"])
                obj = parse_binding(r["o"])
                if obj is None:
                    continue
                expanded_g.add((subj, WDT[pid], obj))
                total += 1
            time.sleep(0.5)

        time.sleep(RATE_LIMIT_DELAY)
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"    Enriched {min(i+BATCH_SIZE, len(new_list))}/{len(new_list)} new entities (+{total} triples)")

    return total


def clean_graph(g):
    """Remove unwanted triples from the expanded graph."""
    print("\nCleaning expanded graph...")
    before = len(g)

    to_remove = []
    for s, p, o in g:
        # remove blank nodes
        if isinstance(s, BNode) or isinstance(o, BNode):
            to_remove.append((s, p, o))
            continue

        # remove statement/qualifier nodes
        p_str = str(p)
        if any(x in p_str for x in ["/prop/statement/", "/prop/qualifier/", "/prop/reference/"]):
            to_remove.append((s, p, o))
            continue

        # remove boring predicates
        if is_boring(p):
            to_remove.append((s, p, o))
            continue

        # remove triples where object is a wikimedia commons file
        if isinstance(o, URIRef):
            o_str = str(o)
            if "commons.wikimedia.org" in o_str or "upload.wikimedia.org" in o_str:
                to_remove.append((s, p, o))
                continue

    for triple in to_remove:
        g.remove(triple)

    after = len(g)
    print(f"  Removed {before - after} triples ({before} -> {after})")
    return g


def compute_stats(g):
    """Calculate various statistics about the knowledge graph."""
    subjects = set()
    predicates = set()
    objects_uri = set()
    pred_counter = Counter()

    for s, p, o in g:
        subjects.add(str(s))
        predicates.add(str(p))
        pred_counter[str(p)] += 1
        if isinstance(o, URIRef):
            objects_uri.add(str(o))

    all_entities = subjects | objects_uri

    # get entity type distribution (rough estimate based on predicates)
    type_dist = Counter()
    for s, p, o in g.triples((None, RDF.type, None)):
        type_dist[str(o)] += 1

    # top 10 predicates
    top_preds = []
    for pred, cnt in pred_counter.most_common(10):
        # make it more readable
        short = pred.split("/")[-1] if "/" in pred else pred
        top_preds.append({"predicate": pred, "short_name": short, "count": cnt})

    stats = {
        "total_triples": len(g),
        "unique_entities": len(all_entities),
        "unique_subjects": len(subjects),
        "unique_predicates": len(predicates),
        "top_10_predicates": top_preds,
        "entity_type_distribution": dict(type_dist.most_common(20)),
    }
    return stats


def main():
    print("=" * 60)
    print("Premier League KG - Wikidata Expansion (v2 - batch mode)")
    print("=" * 60)

    # Load alignment data
    alignment_path = KG_ARTIFACTS / "alignment.ttl"
    mapping_path = DATA_DIR / "entity_mapping.json"

    if not alignment_path.exists():
        print(f"ERROR: alignment file not found at {alignment_path}")
        print("Run alignment.py first!")
        return

    if not mapping_path.exists():
        print(f"ERROR: entity mapping not found at {mapping_path}")
        return

    print("\nLoading alignment data...")
    alignment_g = Graph()
    alignment_g.parse(str(alignment_path), format="turtle")
    print(f"  Loaded {len(alignment_g)} alignment triples")

    with open(mapping_path) as f:
        entity_mapping = json.load(f)
    print(f"  Loaded {len(entity_mapping)} entity mappings")

    # collect all QIDs to expand, grouped by type
    all_qids = []
    team_qids = []
    player_qids = []
    for entity_uri, info in entity_mapping.items():
        qid = info["wikidata_qid"]
        confidence = info.get("confidence", 0)
        if confidence >= 0.7:
            etype = info.get("type", "")
            all_qids.append((qid, info.get("local_name", ""), etype))
            if etype == "team":
                team_qids.append(qid)
            elif etype == "player":
                player_qids.append(qid)

    known_qids = set(q for q, _, _ in all_qids)
    print(f"\n{len(all_qids)} entities to expand ({len(team_qids)} teams, {len(player_qids)} players)")

    # create expanded graph
    expanded_g = Graph()
    expanded_g.bind("wd", WD)
    expanded_g.bind("wdt", WDT)

    sparql = make_sparql()

    # --- Strategy 1: Batch 1-hop expansion for ALL entities ---
    print("\n--- Strategy 1: Batch 1-hop expansion ---")
    all_qid_list = [q for q, _, _ in all_qids]
    for i in range(0, len(all_qid_list), BATCH_SIZE):
        batch = all_qid_list[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(all_qid_list) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  [batch {batch_num}/{total_batches}] Querying {len(batch)} entities...", end=" ")
        count = expand_batch(sparql, batch, expanded_g)
        print(f"+{count} triples (total: {len(expanded_g)})")
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nAfter batch 1-hop: {len(expanded_g)} triples")

    # --- Strategy 2: Reverse expansion for teams ---
    print("\n--- Strategy 2: Reverse expansion (who references our entities) ---")
    # batch reverse queries for teams (most important - pulls in players)
    for i in range(0, len(team_qids), BATCH_SIZE):
        batch = team_qids[i:i+BATCH_SIZE]
        print(f"  [teams batch {i//BATCH_SIZE + 1}] Reverse query for {len(batch)} teams...", end=" ")
        count = expand_reverse(sparql, batch, expanded_g)
        print(f"+{count} triples (total: {len(expanded_g)})")
        time.sleep(RATE_LIMIT_DELAY)

    # also do reverse for players (who else plays for the same teams, etc)
    for i in range(0, len(player_qids), BATCH_SIZE):
        batch = player_qids[i:i+BATCH_SIZE]
        print(f"  [players batch {i//BATCH_SIZE + 1}] Reverse query for {len(batch)} players...", end=" ")
        count = expand_reverse(sparql, batch, expanded_g)
        print(f"+{count} triples (total: {len(expanded_g)})")
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nAfter reverse expansion: {len(expanded_g)} triples")

    # --- Strategy 3: Predicate-controlled batch for teams ---
    print("\n--- Strategy 3: Predicate-controlled batch expansion ---")
    count = expand_predicate_batch(sparql, team_qids, expanded_g)
    print(f"  Predicate batch added {count} triples (total: {len(expanded_g)})")

    # --- Strategy 4: 2-hop expansion for ALL teams ---
    print("\n--- Strategy 4: 2-hop expansion for teams ---")
    for i in range(0, len(team_qids), 5):
        batch = team_qids[i:i+5]
        batch_num = i // 5 + 1
        total_batches = (len(team_qids) + 4) // 5
        print(f"  [batch {batch_num}/{total_batches}] 2-hop for {len(batch)} teams...", end=" ")
        count = expand_2hop_batch(sparql, batch, expanded_g)
        print(f"+{count} triples (total: {len(expanded_g)})")
        time.sleep(RATE_LIMIT_DELAY)

        if len(expanded_g) > 200000:
            print("  Hit 200k cap, stopping 2-hop")
            break

    print(f"\nAfter 2-hop: {len(expanded_g)} triples")

    # --- Strategy 5: Enrich newly discovered entities ---
    print("\n--- Strategy 5: Enrich new entities ---")
    count = enrich_new_entities(sparql, expanded_g, known_qids)
    print(f"  Enrichment added {count} triples (total: {len(expanded_g)})")

    # Clean the data
    expanded_g = clean_graph(expanded_g)

    # Load and merge with original KG
    print("\nMerging with original knowledge graph...")
    original_kg_path = KG_ARTIFACTS / "football_kg.ttl"
    if original_kg_path.exists():
        original_g = Graph()
        original_g.parse(str(original_kg_path), format="turtle")
        print(f"  Original KG: {len(original_g)} triples")

        for triple in original_g:
            expanded_g.add(triple)

        # also add alignment triples
        for triple in alignment_g:
            expanded_g.add(triple)

        print(f"  Merged KG: {len(expanded_g)} triples")
    else:
        print("  WARNING: original KG not found, saving expanded data only")

    # save expanded KB
    os.makedirs(KG_ARTIFACTS, exist_ok=True)
    output_path = KG_ARTIFACTS / "expanded.nt"
    print(f"\nSaving expanded KB to {output_path}...")
    expanded_g.serialize(str(output_path), format="nt")
    print(f"  Done! {len(expanded_g)} triples saved")

    # compute and save stats
    print("\nComputing statistics...")
    stats = compute_stats(expanded_g)

    stats_path = KG_ARTIFACTS / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # print final report
    print("\n" + "=" * 50)
    print("EXPANSION COMPLETE - FINAL STATISTICS")
    print("=" * 50)
    print(f"Total triples:       {stats['total_triples']:,}")
    print(f"Unique entities:     {stats['unique_entities']:,}")
    print(f"Unique predicates:   {stats['unique_predicates']:,}")
    print(f"\nTop 10 predicates:")
    for p in stats["top_10_predicates"]:
        print(f"  {p['short_name']:40s} {p['count']:,}")

    if stats["entity_type_distribution"]:
        print(f"\nEntity type distribution:")
        for t, c in stats["entity_type_distribution"].items():
            short_t = t.split("/")[-1] if "/" in t else t
            print(f"  {short_t}: {c}")

    print(f"\nFiles written:")
    print(f"  {output_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
