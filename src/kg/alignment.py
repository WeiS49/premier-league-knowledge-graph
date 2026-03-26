"""
Entity and predicate alignment with Wikidata.
Maps our local football KG entities to Wikidata QIDs using the search API,
and aligns predicates to Wikidata properties.
"""

import json
import os
import time
from pathlib import Path

import requests
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

# namespaces
FB = Namespace("http://example.org/football/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KG_ARTIFACTS = PROJECT_ROOT / "kg_artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def search_wikidata(name, entity_type=None):
    """Search wikidata for an entity by name. Returns (qid, label, confidence) or None."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json",
        "limit": 5,
    }

    try:
        headers = {"User-Agent": "PremierLeagueKG/1.0 (ESILV student project; contact: student@esilv.fr)"}
        resp = requests.get(url, params=params, timeout=10, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] API error for '{name}': {e}")
        return None

    if not data.get("search"):
        return None

    # try to pick the best match
    results = data["search"]

    # if we have entity_type hint, try to find a better match
    for r in results:
        desc = r.get("description", "").lower()
        qid = r["id"]
        label = r.get("label", "")

        # simple heuristic matching based on type
        if entity_type == "team":
            if any(kw in desc for kw in ["football club", "association football", "soccer"]):
                confidence = 0.95
                return (qid, label, confidence)
        elif entity_type == "player":
            if any(kw in desc for kw in ["footballer", "soccer player", "football player"]):
                confidence = 0.9
                return (qid, label, confidence)
        elif entity_type == "stadium":
            if any(kw in desc for kw in ["stadium", "ground", "arena"]):
                confidence = 0.85
                return (qid, label, confidence)
        elif entity_type == "country":
            if any(kw in desc for kw in ["country", "sovereign", "nation"]):
                confidence = 0.95
                return (qid, label, confidence)

    # fallback: just take the first result
    first = results[0]
    return (first["id"], first.get("label", ""), 0.6)


def get_entity_type(g, entity_uri):
    """Figure out what type an entity is based on its rdf:type or predicates."""
    for _, _, o in g.triples((entity_uri, RDF.type, None)):
        type_str = str(o).lower()
        if "team" in type_str or "club" in type_str:
            return "team"
        elif "player" in type_str or "person" in type_str:
            return "player"
        elif "stadium" in type_str or "venue" in type_str:
            return "stadium"
        elif "country" in type_str or "nation" in type_str:
            return "country"
        elif "league" in type_str or "competition" in type_str:
            return "league"

    # check predicates as fallback
    for _, p, _ in g.triples((entity_uri, None, None)):
        pred = str(p)
        if "playsFor" in pred or "position" in pred:
            return "player"
        if "hasStadium" in pred or "managedBy" in pred:
            return "team"
        if "capacity" in pred:
            return "stadium"

    return None


def extract_local_name(uri):
    """Get the local name from a URI, e.g. http://example.org/football/Arsenal -> Arsenal"""
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.split("/")[-1]


def make_human_readable(name):
    """Convert URI local name to something searchable.
    e.g. 'Manchester_United' -> 'Manchester United'
    """
    # replace underscores and camelCase
    import re
    name = name.replace("_", " ")
    # insert spaces before capitals (for camelCase like 'ManchesterUnited')
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    return name.strip()


def align_entities(g):
    """Find all entities in the graph and try to match them to Wikidata."""
    print("\n=== Entity Alignment ===")

    # collect all subjects that are URIs in our namespace
    entities = set()
    for s, p, o in g:
        if isinstance(s, URIRef) and str(s).startswith(str(FB)):
            entities.add(s)
        # also check objects that are URIs in our namespace
        if isinstance(o, URIRef) and str(o).startswith(str(FB)):
            entities.add(o)

    print(f"Found {len(entities)} entities in the KG")

    alignment_graph = Graph()
    alignment_graph.bind("fb", FB)
    alignment_graph.bind("wd", WD)
    alignment_graph.bind("owl", OWL)

    entity_mapping = {}
    stats = {"total": len(entities), "matched": 0, "failed": 0, "by_type": {}}

    for i, entity in enumerate(sorted(entities, key=str)):
        local_name = extract_local_name(entity)
        search_name = make_human_readable(local_name)
        etype = get_entity_type(g, entity)

        if etype:
            stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1

        # skip some things that aren't really entities
        if len(search_name) < 2:
            continue

        print(f"  [{i+1}/{len(entities)}] Searching: '{search_name}' (type={etype})...", end=" ")

        result = search_wikidata(search_name, etype)

        if result:
            qid, label, confidence = result
            wd_uri = WD[qid]

            # add owl:sameAs triple
            alignment_graph.add((entity, OWL.sameAs, wd_uri))

            entity_mapping[str(entity)] = {
                "local_name": local_name,
                "wikidata_qid": qid,
                "wikidata_label": label,
                "confidence": confidence,
                "type": etype,
            }

            stats["matched"] += 1
            print(f"-> {qid} ({label}) [conf={confidence}]")
        else:
            stats["failed"] += 1
            print("-> NO MATCH")

        # be nice to the API
        time.sleep(0.5)

    return alignment_graph, entity_mapping, stats


def align_predicates():
    """Map our predicates to Wikidata properties using owl:equivalentProperty."""
    print("\n=== Predicate Alignment ===")

    # hardcoded mapping - these are the standard Wikidata properties
    predicate_map = {
        "playsFor": ("P54", "member of sports team"),
        "managedBy": ("P286", "head coach"),
        "hasStadium": ("P115", "home venue"),
        "locatedIn": ("P17", "country"),
        "nationality": ("P27", "country of citizenship"),
        "birthDate": ("P569", "date of birth"),
        "foundedIn": ("P571", "inception"),
        "competesIn": ("P118", "league"),
        "position": ("P413", "position played on team"),
    }

    alignment_graph = Graph()
    alignment_graph.bind("fb", FB)
    alignment_graph.bind("wdt", WDT)
    alignment_graph.bind("owl", OWL)

    for local_pred, (pid, description) in predicate_map.items():
        local_uri = FB[local_pred]
        wd_uri = WDT[pid]

        alignment_graph.add((local_uri, OWL.equivalentProperty, wd_uri))
        print(f"  :{local_pred} -> wdt:{pid} ({description})")

    print(f"\nAligned {len(predicate_map)} predicates")
    return alignment_graph, predicate_map


def main():
    print("=" * 60)
    print("Premier League Knowledge Graph - Entity & Predicate Alignment")
    print("=" * 60)

    # load the initial KG
    kg_path = KG_ARTIFACTS / "football_kg.ttl"
    if not kg_path.exists():
        print(f"ERROR: Could not find KG at {kg_path}")
        print("Please run the KG construction step first.")
        return

    print(f"\nLoading KG from {kg_path}...")
    g = Graph()
    g.parse(str(kg_path), format="turtle")
    print(f"Loaded {len(g)} triples")

    # entity alignment
    entity_alignment, entity_mapping, entity_stats = align_entities(g)

    # predicate alignment
    pred_alignment, pred_map = align_predicates()

    # merge alignment graphs
    combined = Graph()
    combined.bind("fb", FB)
    combined.bind("wd", WD)
    combined.bind("wdt", WDT)
    combined.bind("owl", OWL)

    for triple in entity_alignment:
        combined.add(triple)
    for triple in pred_alignment:
        combined.add(triple)

    # save outputs
    os.makedirs(KG_ARTIFACTS, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    alignment_path = KG_ARTIFACTS / "alignment.ttl"
    combined.serialize(str(alignment_path), format="turtle")
    print(f"\nSaved alignment graph to {alignment_path} ({len(combined)} triples)")

    mapping_path = DATA_DIR / "entity_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(entity_mapping, f, indent=2)
    print(f"Saved entity mapping to {mapping_path} ({len(entity_mapping)} entries)")

    # print statistics
    print("\n" + "=" * 40)
    print("ALIGNMENT STATISTICS")
    print("=" * 40)
    print(f"Total entities found:  {entity_stats['total']}")
    print(f"Successfully matched:  {entity_stats['matched']}")
    print(f"Failed to match:       {entity_stats['failed']}")
    match_rate = entity_stats['matched'] / max(entity_stats['total'], 1) * 100
    print(f"Match rate:            {match_rate:.1f}%")
    print(f"\nBy entity type:")
    for t, count in sorted(entity_stats["by_type"].items()):
        print(f"  {t}: {count}")
    print(f"\nPredicates aligned:    {len(pred_map)}")
    print(f"Total alignment triples: {len(combined)}")


if __name__ == "__main__":
    main()
