"""
prepare.py - Prepare KGE data from the expanded knowledge base
Loads the .nt file, cleans it, splits into train/valid/test
"""

import json
import os
import random
from collections import Counter

import networkx as nx
import rdflib

SEED = 42
random.seed(SEED)

# paths
INPUT_FILE = "kg_artifacts/expanded.nt"
OUTPUT_DIR = "kge_data"


def load_graph(path):
    """Load the ntriples file into an rdflib graph"""
    print(f"Loading graph from {path}...")
    g = rdflib.Graph()
    g.parse(path, format="nt")
    print(f"  Loaded {len(g)} raw triples")
    return g


def clean_triples(g):
    """Clean the graph for embedding training"""

    # step 1: collect all triples as tuples
    triples = set()  # use set to remove duplicates automatically
    for s, p, o in g:
        triples.add((str(s), str(p), str(o)))

    print(f"After dedup: {len(triples)} triples")

    # step 2: remove triples with literal objects
    # literals are things like strings, numbers, dates - not entities
    cleaned = set()
    removed_literals = 0
    for s, p, o in triples:
        # check if object is a literal (not a URI)
        # URIs start with http
        if o.startswith("http://") or o.startswith("https://"):
            cleaned.add((s, p, o))
        else:
            removed_literals += 1

    print(f"Removed {removed_literals} triples with literal objects")
    print(f"  Remaining: {len(cleaned)} triples")
    triples = cleaned

    # step 3: remove blank nodes
    cleaned = set()
    removed_blank = 0
    for s, p, o in triples:
        if "_:" in s or "_:" in o:
            removed_blank += 1
        else:
            cleaned.add((s, p, o))

    print(f"Removed {removed_blank} triples with blank nodes")
    triples = cleaned

    # step 4: remove rare predicates (less than 5 occurrences)
    pred_counts = Counter()
    for s, p, o in triples:
        pred_counts[p] += 1

    rare_preds = {p for p, c in pred_counts.items() if c < 5}
    print(f"Found {len(rare_preds)} rare predicates (< 5 occurrences)")

    cleaned = set()
    for s, p, o in triples:
        if p not in rare_preds:
            cleaned.add((s, p, o))

    print(f"  After removing rare predicates: {len(cleaned)} triples")
    triples = cleaned

    # step 5: remove isolated entities (only 1 connection)
    entity_counts = Counter()
    for s, p, o in triples:
        entity_counts[s] += 1
        entity_counts[o] += 1

    isolated = {e for e, c in entity_counts.items() if c <= 1}
    print(f"Found {len(isolated)} isolated entities")

    cleaned = set()
    for s, p, o in triples:
        if s not in isolated and o not in isolated:
            cleaned.add((s, p, o))

    print(f"  After removing isolated: {len(cleaned)} triples")
    triples = cleaned

    # step 6: keep largest connected component
    print("Finding largest connected component...")
    G = nx.Graph()
    for s, p, o in triples:
        G.add_edge(s, o)

    if nx.is_connected(G):
        print("  Graph is already connected!")
    else:
        components = list(nx.connected_components(G))
        components.sort(key=len, reverse=True)
        print(f"  Found {len(components)} components")
        print(f"  Largest component: {len(components[0])} nodes")

        # keep only triples where both entities are in largest component
        largest = components[0]
        cleaned = set()
        for s, p, o in triples:
            if s in largest and o in largest:
                cleaned.add((s, p, o))

        print(f"  After keeping largest component: {len(cleaned)} triples")
        triples = cleaned

    return list(triples)


def create_mappings(triples):
    """Create entity2id and relation2id mappings"""
    entities = set()
    relations = set()

    for s, p, o in triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)

    # sort for deterministic ordering
    entities = sorted(entities)
    relations = sorted(relations)

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    print(f"\nMappings created:")
    print(f"  {len(entity2id)} unique entities")
    print(f"  {len(relation2id)} unique relations")

    return entity2id, relation2id


def split_data(triples, entity2id):
    """Split into 80/10/10 train/valid/test
    Making sure every entity in valid/test also appears in train
    """
    random.shuffle(triples)

    # first pass: try simple 80/10/10 split
    n = len(triples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train = triples[:n_train]
    valid = triples[n_train : n_train + n_valid]
    test = triples[n_train + n_valid :]

    # check which entities appear in train
    train_entities = set()
    for s, p, o in train:
        train_entities.add(s)
        train_entities.add(o)

    # move triples from valid/test to train if they contain unseen entities
    final_valid = []
    moved_to_train = 0
    for triple in valid:
        s, p, o = triple
        if s not in train_entities or o not in train_entities:
            train.append(triple)
            train_entities.add(s)
            train_entities.add(o)
            moved_to_train += 1
        else:
            final_valid.append(triple)

    final_test = []
    for triple in test:
        s, p, o = triple
        if s not in train_entities or o not in train_entities:
            train.append(triple)
            train_entities.add(s)
            train_entities.add(o)
            moved_to_train += 1
        else:
            final_test.append(triple)

    print(f"\nSplit statistics:")
    print(f"  Moved {moved_to_train} triples to train (entity coverage)")
    print(f"  Train: {len(train)} triples")
    print(f"  Valid: {len(final_valid)} triples")
    print(f"  Test:  {len(final_test)} triples")

    return train, final_valid, final_test


def save_split(triples, filepath):
    """Save triples to tab-separated file: head\trelation\ttail"""
    with open(filepath, "w") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
    print(f"  Saved {len(triples)} triples to {filepath}")


def create_subsamples(train, output_dir):
    """Create subsampled training sets for size sensitivity experiments"""
    if len(train) >= 20000:
        sample_20k = random.sample(train, 20000)
        save_split(sample_20k, os.path.join(output_dir, "train_20k.txt"))
    else:
        print(f"  WARNING: train set too small for 20k subsample ({len(train)} triples)")
        # just save whatever we have
        save_split(train, os.path.join(output_dir, "train_20k.txt"))

    if len(train) >= 50000:
        sample_50k = random.sample(train, 50000)
        save_split(sample_50k, os.path.join(output_dir, "train_50k.txt"))
    else:
        print(f"  WARNING: train set too small for 50k subsample ({len(train)} triples)")
        save_split(train, os.path.join(output_dir, "train_50k.txt"))


def print_final_stats(train, valid, test, entity2id, relation2id):
    """Print final statistics"""
    print("\n" + "=" * 50)
    print("FINAL STATISTICS")
    print("=" * 50)
    print(f"Total triples: {len(train) + len(valid) + len(test)}")
    print(f"  Train: {len(train)}")
    print(f"  Valid: {len(valid)}")
    print(f"  Test:  {len(test)}")
    print(f"Unique entities:  {len(entity2id)}")
    print(f"Unique relations: {len(relation2id)}")
    print()

    # show some example relations
    print("Sample relations:")
    for i, rel in enumerate(list(relation2id.keys())[:10]):
        print(f"  {i+1}. {rel}")


def main():
    # load
    g = load_graph(INPUT_FILE)

    # clean
    print("\n--- Cleaning triples ---")
    triples = clean_triples(g)

    # create mappings
    entity2id, relation2id = create_mappings(triples)

    # split
    train, valid, test = split_data(triples, entity2id)

    # save everything
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nSaving splits...")
    save_split(train, os.path.join(OUTPUT_DIR, "train.txt"))
    save_split(valid, os.path.join(OUTPUT_DIR, "valid.txt"))
    save_split(test, os.path.join(OUTPUT_DIR, "test.txt"))

    # save mappings
    with open(os.path.join(OUTPUT_DIR, "entity2id.json"), "w") as f:
        json.dump(entity2id, f, indent=2)
    print(f"  Saved entity2id.json ({len(entity2id)} entities)")

    with open(os.path.join(OUTPUT_DIR, "relation2id.json"), "w") as f:
        json.dump(relation2id, f, indent=2)
    print(f"  Saved relation2id.json ({len(relation2id)} relations)")

    # subsamples for experiments
    print("\nCreating subsamples...")
    create_subsamples(train, OUTPUT_DIR)

    # print stats
    print_final_stats(train, valid, test, entity2id, relation2id)

    print("\nDone! Data ready for KGE training.")


if __name__ == "__main__":
    main()
