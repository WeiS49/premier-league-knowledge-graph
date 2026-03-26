"""
train_eval.py - Train and evaluate KGE models (TransE, ComplEx) using PyKEEN
Also does nearest neighbor analysis, t-SNE visualization, and rule comparison
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pykeen.models import TransE, ComplEx
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen.losses import MarginRankingLoss
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine


def load_triples(filepath):
    """Load triples from tab-separated file"""
    triples = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append(parts)
    return triples


def make_triples_factory(triples, entity_to_id=None, relation_to_id=None):
    """Convert list of triples to PyKEEN TriplesFactory"""
    # convert to numpy array
    arr = np.array(triples, dtype=str)
    tf = TriplesFactory.from_labeled_triples(
        arr,
        create_inverse_triples=False,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    return tf


def train_model(model_name, train_tf, valid_tf, embedding_dim, num_epochs, batch_size, lr):
    """Train a single KGE model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    try:
        # create model
        if model_name.lower() == "transe":
            model = TransE(
                triples_factory=train_tf,
                embedding_dim=embedding_dim,
                loss=MarginRankingLoss(margin=1.0),
                random_seed=42,
            )
        elif model_name.lower() == "complex":
            model = ComplEx(
                triples_factory=train_tf,
                embedding_dim=embedding_dim,
                random_seed=42,
            )
        else:
            print(f"Unknown model: {model_name}")
            return None, None

        # setup optimizer
        optimizer = torch.optim.Adam(model.get_grad_params(), lr=lr)

        # training loop
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=train_tf,
            optimizer=optimizer,
            negative_sampler_kwargs=dict(
                num_negs_per_pos=1,
            ),
        )

        # train
        print("  Starting training...")
        losses = training_loop.train(
            triples_factory=train_tf,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_tqdm=True,
        )

        print(f"  Training done! Final loss: {losses[-1]:.4f}")
        return model, losses

    except Exception as e:
        print(f"  ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def evaluate_model(model, test_tf, train_tf):
    """Evaluate model on test set using filtered ranking"""
    print("\n  Evaluating...")
    evaluator = RankBasedEvaluator()

    try:
        results = evaluator.evaluate(
            model=model,
            mapped_triples=test_tf.mapped_triples,
            additional_filter_triples=[train_tf.mapped_triples],
            use_tqdm=True,
        )

        # extract metrics
        mrr = results.get_metric("both.realistic.inverse_harmonic_mean_rank")
        hits1 = results.get_metric("both.realistic.hits_at_1")
        hits3 = results.get_metric("both.realistic.hits_at_3")
        hits10 = results.get_metric("both.realistic.hits_at_10")

        print(f"\n  Results:")
        print(f"    MRR:     {mrr:.4f}")
        print(f"    Hits@1:  {hits1:.4f}")
        print(f"    Hits@3:  {hits3:.4f}")
        print(f"    Hits@10: {hits10:.4f}")

        return {"mrr": mrr, "hits1": hits1, "hits3": hits3, "hits10": hits10}

    except Exception as e:
        print(f"  ERROR during evaluation: {e}")
        return None


def size_sensitivity(data_dir, embedding_dim, num_epochs, batch_size, lr):
    """Train TransE on different dataset sizes and compare"""
    print("\n" + "=" * 60)
    print("SIZE SENSITIVITY EXPERIMENT")
    print("=" * 60)

    sizes = ["train_20k.txt", "train_50k.txt", "train.txt"]
    size_labels = ["20k", "50k", "full"]
    all_results = {}

    # load test set (same for all)
    test_triples = load_triples(os.path.join(data_dir, "test.txt"))

    for size_file, label in zip(sizes, size_labels):
        filepath = os.path.join(data_dir, size_file)
        if not os.path.exists(filepath):
            print(f"  Skipping {label} - file not found")
            continue

        print(f"\n--- Training on {label} dataset ---")
        train_triples = load_triples(filepath)
        print(f"  {len(train_triples)} training triples")

        # make factories
        train_tf = make_triples_factory(train_triples)

        # filter test triples to only include known entities/relations
        known_entities = set()
        known_relations = set()
        for s, p, o in train_triples:
            known_entities.add(s)
            known_entities.add(o)
            known_relations.add(p)

        filtered_test = [
            t for t in test_triples
            if t[0] in known_entities and t[2] in known_entities and t[1] in known_relations
        ]
        print(f"  {len(filtered_test)} test triples (after filtering)")

        if len(filtered_test) < 10:
            print("  Too few test triples, skipping")
            continue

        test_tf = make_triples_factory(
            filtered_test,
            entity_to_id=train_tf.entity_to_id,
            relation_to_id=train_tf.relation_to_id,
        )

        # train with fewer epochs for subsamples (save time)
        epochs = min(num_epochs, 50)
        model, losses = train_model("TransE", train_tf, None, embedding_dim, epochs, batch_size, lr)

        if model is not None:
            metrics = evaluate_model(model, test_tf, train_tf)
            if metrics:
                all_results[label] = metrics

    # print comparison table
    if all_results:
        print("\n\n" + "=" * 60)
        print("SIZE SENSITIVITY COMPARISON")
        print("=" * 60)
        print(f"{'Size':<10} {'MRR':<10} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10}")
        print("-" * 50)
        for label in size_labels:
            if label in all_results:
                r = all_results[label]
                print(f"{label:<10} {r['mrr']:<10.4f} {r['hits1']:<10.4f} {r['hits3']:<10.4f} {r['hits10']:<10.4f}")


def nearest_neighbors(model, train_tf, n_neighbors=5):
    """Find nearest neighbors for selected entities in embedding space"""
    print("\n" + "=" * 60)
    print("NEAREST NEIGHBOR ANALYSIS")
    print("=" * 60)

    # get entity embeddings - ComplEx uses complex numbers, take real part
    entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
    if np.iscomplexobj(entity_embeddings):
        entity_embeddings = entity_embeddings.real
    id_to_entity = {v: k for k, v in train_tf.entity_to_id.items()}

    # select some entities to analyze - use Wikidata QIDs
    target_entities = [
        "http://www.wikidata.org/entity/Q9617",    # Arsenal
        "http://www.wikidata.org/entity/Q18741",   # Manchester City
        "http://www.wikidata.org/entity/Q9616",    # Chelsea
        "http://www.wikidata.org/entity/Q5794",    # Tottenham
        "http://www.wikidata.org/entity/Q19568",   # Brentford
    ]

    for entity_uri in target_entities:
        if entity_uri not in train_tf.entity_to_id:
            # try shorter name
            short_name = entity_uri.split("/")[-1].replace("_", " ")
            print(f"\n  Entity not found: {short_name}")
            continue

        entity_id = train_tf.entity_to_id[entity_uri]
        entity_emb = entity_embeddings[entity_id]

        # compute distances to all other entities
        distances = []
        for i in range(len(entity_embeddings)):
            if i == entity_id:
                continue
            dist = np.linalg.norm(entity_emb - entity_embeddings[i])
            distances.append((i, dist))

        # sort by distance
        distances.sort(key=lambda x: x[1])

        short_name = entity_uri.split("/")[-1].replace("_", " ")
        print(f"\n  Nearest neighbors of '{short_name}':")
        for i, (idx, dist) in enumerate(distances[:n_neighbors]):
            neighbor_uri = id_to_entity[idx]
            neighbor_name = neighbor_uri.split("/")[-1].replace("_", " ")
            print(f"    {i+1}. {neighbor_name} (dist: {dist:.4f})")


def tsne_visualization(model, train_tf, output_path):
    """Create t-SNE visualization of entity embeddings"""
    print("\n" + "=" * 60)
    print("t-SNE VISUALIZATION")
    print("=" * 60)

    # get embeddings - ComplEx uses complex numbers, take real part for viz
    entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
    if np.iscomplexobj(entity_embeddings):
        entity_embeddings = entity_embeddings.real
    id_to_entity = {v: k for k, v in train_tf.entity_to_id.items()}

    print(f"  Running t-SNE on {len(entity_embeddings)} entities...")

    # subsample if too many entities (t-SNE is slow on large datasets)
    max_points = 3000
    if len(entity_embeddings) > max_points:
        print(f"  Subsampling to {max_points} entities for visualization")
        indices = np.random.choice(len(entity_embeddings), max_points, replace=False)
        embeddings_sub = entity_embeddings[indices]
    else:
        indices = np.arange(len(entity_embeddings))
        embeddings_sub = entity_embeddings

    # run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_sub)

    # try to color by entity type
    # we'll use a simple heuristic: check if URI contains certain keywords
    colors = []
    for idx in indices:
        uri = id_to_entity[idx]
        if "F.C." in uri or "United" in uri or "City" in uri or "FC" in uri:
            colors.append("red")  # clubs
        elif "League" in uri or "Cup" in uri or "Championship" in uri:
            colors.append("blue")  # competitions
        elif "Stadium" in uri or "Park" in uri or "Ground" in uri:
            colors.append("green")  # venues
        else:
            colors.append("gray")  # other (probably players/managers)

    # plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5, s=10)

    # add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Clubs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Competitions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Venues'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("t-SNE Visualization of Entity Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved t-SNE plot to {output_path}")


def rule_vs_embedding_comparison(model, train_tf):
    """Compare rule-based reasoning with embedding arithmetic
    Idea: in football, playsFor(player, club) + hasLeague(club, league) ~ playsInLeague(player, league)
    So we check if vec(playsFor) + vec(inLeague) ≈ vec(playsInLeague) or similar
    """
    print("\n" + "=" * 60)
    print("RULE-BASED vs EMBEDDING COMPARISON")
    print("=" * 60)

    # get relation embeddings
    relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()
    id_to_relation = {v: k for k, v in train_tf.relation_to_id.items()}

    print("\n  Available relations:")
    for rid, rname in sorted(id_to_relation.items()):
        short = rname.split("/")[-1] if "/" in rname else rname
        print(f"    [{rid}] {short}")

    # try to find relevant relations for football domain
    # look for relations like: team, league, plays, position etc
    relation_names = {v: k for k, v in train_tf.relation_to_id.items()}

    # try different combinations based on what's available
    # rule: playsFor + league_relation ≈ playsInLeague
    combos_to_try = [
        # (rel_a, rel_b, expected_result, description)
        ("team", "league", "playsInLeague", "playsFor + inLeague ≈ playsInLeague"),
        ("club", "league", "competition", "club + league ≈ competition"),
    ]

    # find relations containing keywords
    def find_relation(keyword):
        for rname, rid in train_tf.relation_to_id.items():
            if keyword.lower() in rname.lower():
                return rname, rid
        return None, None

    # just compare all pairs of relations for cosine similarity
    print("\n  Relation similarity matrix (top pairs):")
    similarities = []
    rel_list = list(id_to_relation.items())
    for i in range(len(rel_list)):
        for j in range(i + 1, len(rel_list)):
            id_i, name_i = rel_list[i]
            id_j, name_j = rel_list[j]
            sim = 1 - cosine(relation_embeddings[id_i], relation_embeddings[id_j])
            short_i = name_i.split("/")[-1] if "/" in name_i else name_i
            short_j = name_j.split("/")[-1] if "/" in name_j else name_j
            similarities.append((short_i, short_j, sim))

    # sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 10 most similar relation pairs:")
    for i, (r1, r2, sim) in enumerate(similarities[:10]):
        print(f"    {i+1}. {r1} <-> {r2}: {sim:.4f}")

    print(f"\n  Bottom 5 least similar relation pairs:")
    for i, (r1, r2, sim) in enumerate(similarities[-5:]):
        print(f"    {i+1}. {r1} <-> {r2}: {sim:.4f}")

    # try vector arithmetic: for each triple of relations, check if a + b ≈ c
    if len(rel_list) >= 3:
        print(f"\n  Relation arithmetic (checking if vec(A) + vec(B) ≈ vec(C)):")
        best_arithmetic = []
        for i in range(min(len(rel_list), 15)):
            for j in range(min(len(rel_list), 15)):
                if i == j:
                    continue
                combined = relation_embeddings[rel_list[i][0]] + relation_embeddings[rel_list[j][0]]
                for k in range(min(len(rel_list), 15)):
                    if k == i or k == j:
                        continue
                    target = relation_embeddings[rel_list[k][0]]
                    sim = 1 - cosine(combined, target)
                    short_i = rel_list[i][1].split("/")[-1]
                    short_j = rel_list[j][1].split("/")[-1]
                    short_k = rel_list[k][1].split("/")[-1]
                    best_arithmetic.append((short_i, short_j, short_k, sim))

        best_arithmetic.sort(key=lambda x: x[3], reverse=True)
        print(f"\n  Top 5 relation analogies:")
        for a, b, c, sim in best_arithmetic[:5]:
            print(f"    vec({a}) + vec({b}) ≈ vec({c})  (cosine sim: {sim:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate KGE models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--model", type=str, default="both", choices=["transe", "complex", "both"],
                        help="Which model to train")
    parser.add_argument("--data-dir", type=str, default="kge_data/", help="Directory with train/valid/test splits")
    args = parser.parse_args()

    print("=" * 60)
    print("KGE Training and Evaluation")
    print("=" * 60)
    print(f"  Data dir: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Embedding dim: {args.dim}")
    print(f"  Model: {args.model}")

    # load data
    print("\nLoading data...")
    train_triples = load_triples(os.path.join(args.data_dir, "train.txt"))
    valid_triples = load_triples(os.path.join(args.data_dir, "valid.txt"))
    test_triples = load_triples(os.path.join(args.data_dir, "test.txt"))

    print(f"  Train: {len(train_triples)} triples")
    print(f"  Valid: {len(valid_triples)} triples")
    print(f"  Test:  {len(test_triples)} triples")

    # create triples factories
    train_tf = make_triples_factory(train_triples)
    valid_tf = make_triples_factory(
        valid_triples,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    test_tf = make_triples_factory(
        test_triples,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )

    # hyperparameters
    batch_size = 256
    lr = 0.01

    # store results for comparison
    results = {}
    trained_models = {}

    # train models
    models_to_train = []
    if args.model == "both":
        models_to_train = ["TransE", "ComplEx"]
    elif args.model == "transe":
        models_to_train = ["TransE"]
    else:
        models_to_train = ["ComplEx"]

    for model_name in models_to_train:
        model, losses = train_model(model_name, train_tf, valid_tf, args.dim, args.epochs, batch_size, lr)

        if model is not None:
            # evaluate
            metrics = evaluate_model(model, test_tf, train_tf)
            if metrics:
                results[model_name] = metrics
                trained_models[model_name] = model

            # save model
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"{model_name.lower()}_model.pkl")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved model to {model_path}")

    # print comparison if we trained both
    if len(results) >= 2:
        print("\n\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<15} {'MRR':<10} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10}")
        print("-" * 55)
        for model_name, r in results.items():
            print(f"{model_name:<15} {r['mrr']:<10.4f} {r['hits1']:<10.4f} {r['hits3']:<10.4f} {r['hits10']:<10.4f}")

    # pick best model for further analysis
    best_model_name = max(results, key=lambda k: results[k]["mrr"]) if results else None
    best_model = trained_models.get(best_model_name) if best_model_name else None

    if best_model is not None:
        print(f"\nBest model: {best_model_name} (MRR: {results[best_model_name]['mrr']:.4f})")

        # nearest neighbor analysis
        nearest_neighbors(best_model, train_tf)

        # t-SNE visualization
        tsne_visualization(best_model, train_tf, "reports/tsne_embeddings.png")

        # rule vs embedding comparison
        try:
            rule_vs_embedding_comparison(best_model, train_tf)
        except Exception as e:
            print(f"\n  Rule comparison failed: {e}")

    # size sensitivity experiment
    try:
        size_sensitivity(args.data_dir, args.dim, args.epochs, batch_size, lr)
    except Exception as e:
        print(f"\n  Size sensitivity experiment failed: {e}")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
