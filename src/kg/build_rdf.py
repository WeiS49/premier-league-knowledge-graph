"""
Build RDF Knowledge Graph for Premier League data
Creates triples about teams, players, stadiums etc.
"""

import json
import os
import re
from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD, URIRef

# setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAMS_PATH = os.path.join(BASE_DIR, "data", "processed", "teams_clean.json")
PLAYERS_PATH = os.path.join(BASE_DIR, "data", "processed", "players_clean.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "kg_artifacts")

# our namespace
FB = Namespace("http://example.org/football/")


def make_uri(name):
    """Convert a name string to a valid URI fragment
    e.g. 'Bukayo Saka' -> 'Bukayo_Saka'
    """
    # remove special chars and replace spaces with underscores
    clean = re.sub(r'[^\w\s-]', '', name)
    clean = clean.strip().replace(' ', '_')
    return clean


def load_data():
    """Load team and player data from json"""
    with open(TEAMS_PATH, "r") as f:
        teams = json.load(f)
    with open(PLAYERS_PATH, "r") as f:
        players = json.load(f)
    return teams, players


def add_team_triples(g, team):
    """Add RDF triples for a single team"""
    teamName = team.get("name", "")
    if not teamName:
        return

    team_uri = FB[make_uri(teamName)]

    # basic type
    g.add((team_uri, RDF.type, FB["Team"]))
    g.add((team_uri, FB["name"], Literal(teamName)))

    # stadium
    if "stadium" in team and team["stadium"]:
        stadium_uri = FB[make_uri(team["stadium"])]
        g.add((team_uri, FB["hasStadium"], stadium_uri))
        g.add((stadium_uri, RDF.type, FB["Stadium"]))
        g.add((stadium_uri, FB["name"], Literal(team["stadium"])))
        # add capacity if we have it
        if "stadium_capacity" in team:
            g.add((stadium_uri, FB["capacity"], Literal(team["stadium_capacity"], datatype=XSD.integer)))

    # founded year
    if "founded" in team and team["founded"]:
        g.add((team_uri, FB["foundedIn"], Literal(str(team["founded"]))))

    # location
    if "location" in team and team["location"]:
        loc_uri = FB[make_uri(team["location"])]
        g.add((team_uri, FB["locatedIn"], loc_uri))
        g.add((loc_uri, RDF.type, FB["Country"]))  # could be city actually but ok

    # always add Premier League
    g.add((team_uri, FB["competesIn"], FB["Premier_League"]))
    g.add((FB["Premier_League"], RDF.type, FB["League"]))
    g.add((FB["Premier_League"], FB["name"], Literal("Premier League")))

    # manager
    if "manager" in team and team["manager"]:
        manager_uri = FB[make_uri(team["manager"])]
        g.add((team_uri, FB["managedBy"], manager_uri))
        g.add((manager_uri, RDF.type, FB["Manager"]))
        g.add((manager_uri, RDF.type, FB["Person"]))
        g.add((manager_uri, FB["name"], Literal(team["manager"])))

    # honours/trophies
    if "honours" in team:
        for i, honour in enumerate(team["honours"]):
            g.add((team_uri, FB["hasHonour"], Literal(honour)))


def add_player_triples(g, player):
    """Add RDF triples for a single player"""
    playerName = player.get("name", "")
    if not playerName:
        return

    player_uri = FB[make_uri(playerName)]

    # type
    g.add((player_uri, RDF.type, FB["Player"]))
    g.add((player_uri, RDF.type, FB["Person"]))
    g.add((player_uri, FB["name"], Literal(playerName)))

    # team
    if "team" in player and player["team"]:
        team_uri = FB[make_uri(player["team"])]
        g.add((player_uri, FB["playsFor"], team_uri))

    # nationality
    if "nationality" in player and player["nationality"]:
        nat_uri = FB[make_uri(player["nationality"])]
        g.add((player_uri, FB["nationality"], nat_uri))
        g.add((nat_uri, RDF.type, FB["Country"]))
        g.add((nat_uri, FB["name"], Literal(player["nationality"])))

    # position
    if "position" in player and player["position"]:
        g.add((player_uri, FB["position"], Literal(player["position"])))

    # birth date
    if "birth_date" in player and player["birth_date"]:
        g.add((player_uri, FB["birthDate"], Literal(player["birth_date"], datatype=XSD.date)))

    # jersey number
    if "number" in player:
        g.add((player_uri, FB["jerseyNumber"], Literal(player["number"], datatype=XSD.integer)))

    # market value if available
    if "market_value" in player and player["market_value"]:
        g.add((player_uri, FB["marketValue"], Literal(player["market_value"])))


def build_knowledge_graph():
    """Main function to build the KG"""
    print("=== Building Premier League Knowledge Graph ===\n")

    # create graph
    g = Graph()
    g.bind("fb", FB)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # load data
    print("Loading data...")
    teams, players = load_data()
    print(f"  {len(teams)} teams, {len(players)} players")

    # add team triples
    print("\nAdding team triples...")
    for team in teams:
        add_team_triples(g, team)

    # add player triples
    print("Adding player triples...")
    for player in players:
        add_player_triples(g, player)

    # add some extra triples about the league itself
    g.add((FB["Premier_League"], FB["country"], FB["England"]))
    g.add((FB["England"], RDF.type, FB["Country"]))
    g.add((FB["England"], FB["name"], Literal("England")))
    g.add((FB["Premier_League"], FB["foundedIn"], Literal("1992")))
    g.add((FB["Premier_League"], FB["numberOfTeams"], Literal(20, datatype=XSD.integer)))

    # print stats
    print_stats(g)

    # save to files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nt_path = os.path.join(OUTPUT_DIR, "football_kg.nt")
    ttl_path = os.path.join(OUTPUT_DIR, "football_kg.ttl")

    print(f"\nSaving N-Triples to {nt_path}...")
    g.serialize(destination=nt_path, format="nt")

    print(f"Saving Turtle to {ttl_path}...")
    g.serialize(destination=ttl_path, format="turtle")

    print("\nDone! Knowledge graph built successfully.")
    return g


def print_stats(g):
    """Print statistics about the knowledge graph"""
    # count triples
    num_triples = len(g)

    # count unique subjects (entities)
    subjects = set()
    for s, p, o in g:
        subjects.add(s)
        if isinstance(o, URIRef):
            subjects.add(o)

    # count unique predicates (relations)
    predicates = set()
    for s, p, o in g:
        predicates.add(p)

    print(f"\n--- Knowledge Graph Statistics ---")
    print(f"  Total triples:    {num_triples}")
    print(f"  Unique entities:  {len(subjects)}")
    print(f"  Unique relations: {len(predicates)}")

    # count by type
    type_counts = {}
    for s, p, o in g.triples((None, RDF.type, None)):
        type_name = str(o).split("/")[-1]
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print("\n  Entities by type:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # check if we meet the targets
    if num_triples >= 100:
        print(f"\n  [OK] Target met: >= 100 triples ({num_triples})")
    else:
        print(f"\n  [!!] Below target: need >= 100 triples, got {num_triples}")

    if len(subjects) >= 50:
        print(f"  [OK] Target met: >= 50 entities ({len(subjects)})")
    else:
        print(f"  [!!] Below target: need >= 50 entities, got {len(subjects)}")


if __name__ == "__main__":
    build_knowledge_graph()
