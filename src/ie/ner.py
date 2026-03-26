"""
NER extraction for Premier League teams and players
Uses spaCy to find named entities in the cleaned data
"""

import json
import os
import spacy

# load the spacy model
nlp = spacy.load("en_core_web_sm")

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAMS_PATH = os.path.join(BASE_DIR, "data", "processed", "teams_clean.json")
PLAYERS_PATH = os.path.join(BASE_DIR, "data", "processed", "players_clean.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "ner_results.json")
AMBIGUITY_PATH = os.path.join(BASE_DIR, "data", "processed", "ambiguity_cases.json")


def load_data():
    """Load the cleaned json files"""
    with open(TEAMS_PATH, "r") as f:
        teams = json.load(f)
    with open(PLAYERS_PATH, "r") as f:
        players = json.load(f)
    return teams, players


def extract_entities_from_text(text, source_name, source_type):
    """Run spaCy NER on a piece of text and return entities

    We only keep these entity types: PERSON, ORG, GPE, DATE, EVENT
    """
    doc = nlp(text)

    # the entity types we care about
    keepLabels = {"PERSON", "ORG", "GPE", "DATE", "EVENT"}

    entities = []
    for ent in doc.ents:
        if ent.label_ in keepLabels:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": source_name,
                "source_type": source_type
            })

    return entities


def process_teams(teams):
    """Extract entities from all team data"""
    all_entities = []

    for team in teams:
        teamName = team.get("name", "Unknown")
        # combine all text fields for the team
        text_parts = []
        if "description" in team:
            text_parts.append(team["description"])
        if "history" in team:
            text_parts.append(team["history"])
        if "stadium" in team:
            text_parts.append(f"The team plays at {team['stadium']}.")
        if "location" in team:
            text_parts.append(f"Located in {team['location']}.")
        if "manager" in team:
            text_parts.append(f"The current manager is {team['manager']}.")
        if "founded" in team:
            text_parts.append(f"Founded in {team['founded']}.")

        # add some extra context
        if "honours" in team:
            for h in team["honours"]:
                text_parts.append(h)

        combined_text = " ".join(text_parts)

        if combined_text.strip():
            ents = extract_entities_from_text(combined_text, teamName, "team")
            all_entities.extend(ents)
            print(f"  Team '{teamName}': found {len(ents)} entities")

    return all_entities


def process_players(players):
    """Extract entities from all player data"""
    all_entities = []

    for player in players:
        playerName = player.get("name", "Unknown")

        # build text from player info
        text_parts = []
        if "description" in player:
            text_parts.append(player["description"])
        if "biography" in player:
            text_parts.append(player["biography"])
        # add basic info as sentences
        if "nationality" in player:
            text_parts.append(f"{playerName} is from {player['nationality']}.")
        if "team" in player:
            text_parts.append(f"{playerName} plays for {player['team']}.")
        if "birth_date" in player:
            text_parts.append(f"Born on {player['birth_date']}.")
        if "position" in player:
            text_parts.append(f"Plays as {player['position']}.")

        combined_text = " ".join(text_parts)

        if combined_text.strip():
            ents = extract_entities_from_text(combined_text, playerName, "player")
            all_entities.extend(ents)
            print(f"  Player '{playerName}': found {len(ents)} entities")

    return all_entities


def find_ambiguity_cases(all_entities):
    """
    Find cases where entity recognition is ambiguous.
    For example Arsenal could be ORG or GPE, some player names
    could also be place names, etc.
    """
    # group entities by text
    entity_labels = {}
    for ent in all_entities:
        txt = ent["text"]
        if txt not in entity_labels:
            entity_labels[txt] = set()
        entity_labels[txt].add(ent["label"])

    # find entities that got multiple labels
    multi_label = {}
    for txt, labels in entity_labels.items():
        if len(labels) > 1:
            multi_label[txt] = list(labels)

    # hardcoded known ambiguity cases for Premier League
    # (in case spaCy doesn't catch them naturally)
    known_ambiguities = [
        {
            "entity": "Arsenal",
            "possible_labels": ["ORG", "GPE"],
            "explanation": "Arsenal is a football club (ORG) but the word 'arsenal' originally refers to a military weapons depot. SpaCy might confuse it with a location.",
            "type": "org_vs_location"
        },
        {
            "entity": "Jordan Henderson",
            "possible_labels": ["PERSON", "GPE"],
            "explanation": "Jordan is both a common first name and a country (GPE). Henderson is also a place name in several countries.",
            "type": "person_vs_location"
        },
        {
            "entity": "1886",
            "possible_labels": ["DATE", "CARDINAL"],
            "explanation": "The founding year of Arsenal. Without context it could be interpreted as just a number (CARDINAL) rather than a date.",
            "type": "date_vs_number"
        }
    ]

    # add any naturally found ambiguities
    for txt, labels in multi_label.items():
        already_there = any(a["entity"] == txt for a in known_ambiguities)
        if not already_there:
            known_ambiguities.append({
                "entity": txt,
                "possible_labels": labels,
                "explanation": f"SpaCy assigned multiple labels to '{txt}': {', '.join(labels)}",
                "type": "detected_automatically"
            })

    return known_ambiguities[:10]  # keep top 10 at most


def print_examples(entities, n=15):
    """Print some example entities for debugging"""
    print("\n--- Example Extracted Entities ---")
    print(f"{'Text':<25} {'Label':<10} {'Source':<20} {'Type':<10}")
    print("-" * 70)

    # show first n entities
    for ent in entities[:n]:
        print(f"{ent['text']:<25} {ent['label']:<10} {ent['source']:<20} {ent['source_type']:<10}")

    print(f"\nTotal entities extracted: {len(entities)}")

    # count by label
    label_counts = {}
    for ent in entities:
        lbl = ent["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print("\nEntities by type:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt}")


def main():
    print("=== NER Extraction for Premier League Data ===\n")

    # load data
    print("Loading cleaned data...")
    teams, players = load_data()
    print(f"Loaded {len(teams)} teams and {len(players)} players\n")

    # process teams
    print("Processing teams...")
    team_entities = process_teams(teams)

    # process players
    print("\nProcessing players...")
    player_entities = process_players(players)

    # combine all entities
    all_entities = team_entities + player_entities

    # print examples
    print_examples(all_entities)

    # find ambiguity cases
    print("\n\n=== Ambiguity Cases ===")
    ambiguities = find_ambiguity_cases(all_entities)
    for i, amb in enumerate(ambiguities[:3], 1):
        print(f"\n{i}. Entity: '{amb['entity']}'")
        print(f"   Possible labels: {amb['possible_labels']}")
        print(f"   Explanation: {amb['explanation']}")

    # save results
    print(f"\nSaving NER results to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_entities, f, indent=2)

    print(f"Saving ambiguity cases to {AMBIGUITY_PATH}...")
    with open(AMBIGUITY_PATH, "w") as f:
        json.dump(ambiguities, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
