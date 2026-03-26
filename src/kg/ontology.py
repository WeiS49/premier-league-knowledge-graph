"""
Generate OWL Ontology for the Premier League Knowledge Graph
Defines classes, object properties, and data properties
"""

import os
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL, XSD

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "kg_artifacts")

# namespaces
FB = Namespace("http://example.org/football/")


def create_ontology():
    """Build the OWL ontology for our football KG"""

    g = Graph()
    g.bind("fb", FB)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # declare it as an ontology
    ontology_uri = URIRef("http://example.org/football/ontology")
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal("Premier League Football Ontology")))
    g.add((ontology_uri, RDFS.comment, Literal("An ontology for representing Premier League football knowledge")))

    # ==================
    # CLASSES
    # ==================

    print("Defining classes...")

    # Person (top level)
    g.add((FB["Person"], RDF.type, OWL.Class))
    g.add((FB["Person"], RDFS.label, Literal("Person")))

    # Player is subclass of Person
    g.add((FB["Player"], RDF.type, OWL.Class))
    g.add((FB["Player"], RDFS.subClassOf, FB["Person"]))
    g.add((FB["Player"], RDFS.label, Literal("Player")))

    # Manager is subclass of Person
    g.add((FB["Manager"], RDF.type, OWL.Class))
    g.add((FB["Manager"], RDFS.subClassOf, FB["Person"]))
    g.add((FB["Manager"], RDFS.label, Literal("Manager")))

    # Team
    g.add((FB["Team"], RDF.type, OWL.Class))
    g.add((FB["Team"], RDFS.label, Literal("Team")))
    g.add((FB["Team"], RDFS.comment, Literal("A football team/club")))

    # League
    g.add((FB["League"], RDF.type, OWL.Class))
    g.add((FB["League"], RDFS.label, Literal("League")))

    # Stadium
    g.add((FB["Stadium"], RDF.type, OWL.Class))
    g.add((FB["Stadium"], RDFS.label, Literal("Stadium")))

    # Country
    g.add((FB["Country"], RDF.type, OWL.Class))
    g.add((FB["Country"], RDFS.label, Literal("Country")))

    # Competition - more general than League
    g.add((FB["Competition"], RDF.type, OWL.Class))
    g.add((FB["Competition"], RDFS.label, Literal("Competition")))
    g.add((FB["League"], RDFS.subClassOf, FB["Competition"]))  # League is a Competition

    # Award (for wonAward property)
    g.add((FB["Award"], RDF.type, OWL.Class))
    g.add((FB["Award"], RDFS.label, Literal("Award")))

    # ==================
    # OBJECT PROPERTIES
    # ==================

    print("Defining object properties...")

    # playsFor: Player -> Team
    g.add((FB["playsFor"], RDF.type, OWL.ObjectProperty))
    g.add((FB["playsFor"], RDFS.domain, FB["Player"]))
    g.add((FB["playsFor"], RDFS.range, FB["Team"]))
    g.add((FB["playsFor"], RDFS.label, Literal("plays for")))

    # managedBy: Team -> Manager
    g.add((FB["managedBy"], RDF.type, OWL.ObjectProperty))
    g.add((FB["managedBy"], RDFS.domain, FB["Team"]))
    g.add((FB["managedBy"], RDFS.range, FB["Manager"]))
    g.add((FB["managedBy"], RDFS.label, Literal("managed by")))

    # hasStadium: Team -> Stadium
    g.add((FB["hasStadium"], RDF.type, OWL.ObjectProperty))
    g.add((FB["hasStadium"], RDFS.domain, FB["Team"]))
    g.add((FB["hasStadium"], RDFS.range, FB["Stadium"]))
    g.add((FB["hasStadium"], RDFS.label, Literal("has stadium")))

    # locatedIn: Team|Stadium -> Country
    # note: OWL doesn't directly support union domains easily
    # so we just set it without strict domain for now
    g.add((FB["locatedIn"], RDF.type, OWL.ObjectProperty))
    g.add((FB["locatedIn"], RDFS.range, FB["Country"]))
    g.add((FB["locatedIn"], RDFS.label, Literal("located in")))
    g.add((FB["locatedIn"], RDFS.comment, Literal("Domain can be Team or Stadium")))

    # competesIn: Team -> League
    g.add((FB["competesIn"], RDF.type, OWL.ObjectProperty))
    g.add((FB["competesIn"], RDFS.domain, FB["Team"]))
    g.add((FB["competesIn"], RDFS.range, FB["League"]))
    g.add((FB["competesIn"], RDFS.label, Literal("competes in")))

    # nationality: Person -> Country
    g.add((FB["nationality"], RDF.type, OWL.ObjectProperty))
    g.add((FB["nationality"], RDFS.domain, FB["Person"]))
    g.add((FB["nationality"], RDFS.range, FB["Country"]))
    g.add((FB["nationality"], RDFS.label, Literal("has nationality")))

    # wonAward: Person -> Award (optional/extra)
    g.add((FB["wonAward"], RDF.type, OWL.ObjectProperty))
    g.add((FB["wonAward"], RDFS.domain, FB["Person"]))
    g.add((FB["wonAward"], RDFS.range, FB["Award"]))
    g.add((FB["wonAward"], RDFS.label, Literal("won award")))

    # hasPlayer - inverse of playsFor basically
    g.add((FB["hasPlayer"], RDF.type, OWL.ObjectProperty))
    g.add((FB["hasPlayer"], RDFS.domain, FB["Team"]))
    g.add((FB["hasPlayer"], RDFS.range, FB["Player"]))
    g.add((FB["hasPlayer"], OWL.inverseOf, FB["playsFor"]))

    # ==================
    # DATA PROPERTIES
    # ==================

    print("Defining data properties...")

    # name
    g.add((FB["name"], RDF.type, OWL.DatatypeProperty))
    g.add((FB["name"], RDFS.range, XSD.string))
    g.add((FB["name"], RDFS.label, Literal("name")))

    # foundedIn - year as string
    g.add((FB["foundedIn"], RDF.type, OWL.DatatypeProperty))
    g.add((FB["foundedIn"], RDFS.domain, FB["Team"]))
    g.add((FB["foundedIn"], RDFS.range, XSD.string))
    g.add((FB["foundedIn"], RDFS.label, Literal("founded in")))

    # birthDate
    g.add((FB["birthDate"], RDF.type, OWL.DatatypeProperty))
    g.add((FB["birthDate"], RDFS.domain, FB["Person"]))
    g.add((FB["birthDate"], RDFS.range, XSD.date))
    g.add((FB["birthDate"], RDFS.label, Literal("birth date")))

    # position
    g.add((FB["position"], RDF.type, OWL.DatatypeProperty))
    g.add((FB["position"], RDFS.domain, FB["Player"]))
    g.add((FB["position"], RDFS.range, XSD.string))
    g.add((FB["position"], RDFS.label, Literal("position")))

    # capacity (for stadiums)
    g.add((FB["capacity"], RDF.type, OWL.DatatypeProperty))
    g.add((FB["capacity"], RDFS.domain, FB["Stadium"]))
    g.add((FB["capacity"], RDFS.range, XSD.integer))
    g.add((FB["capacity"], RDFS.label, Literal("capacity")))

    # jerseyNumber
    g.add((FB["jerseyNumber"], RDF.type, OWL.DatatypeProperty))
    g.add((FB["jerseyNumber"], RDFS.domain, FB["Player"]))
    g.add((FB["jerseyNumber"], RDFS.range, XSD.integer))

    return g


def save_ontology(g):
    """Save ontology to turtle file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ontology.ttl")

    print(f"\nSaving ontology to {output_path}...")
    g.serialize(destination=output_path, format="turtle")
    print("Saved!")

    return output_path


def print_ontology_stats(g):
    """Print some stats about what we defined"""
    # count classes
    classes = list(g.subjects(RDF.type, OWL.Class))
    obj_props = list(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = list(g.subjects(RDF.type, OWL.DatatypeProperty))

    print(f"\n--- Ontology Statistics ---")
    print(f"  Classes:            {len(classes)}")
    print(f"  Object Properties:  {len(obj_props)}")
    print(f"  Data Properties:    {len(data_props)}")
    print(f"  Total triples:      {len(g)}")

    print("\n  Classes defined:")
    for c in classes:
        label = g.value(c, RDFS.label)
        name = str(c).split("/")[-1]
        print(f"    - {name}" + (f" ({label})" if label else ""))

    print("\n  Object Properties:")
    for p in obj_props:
        domain = g.value(p, RDFS.domain)
        range_ = g.value(p, RDFS.range)
        pname = str(p).split("/")[-1]
        d = str(domain).split("/")[-1] if domain else "?"
        r = str(range_).split("/")[-1] if range_ else "?"
        print(f"    - {pname}: {d} -> {r}")

    print("\n  Data Properties:")
    for p in data_props:
        pname = str(p).split("/")[-1]
        range_ = g.value(p, RDFS.range)
        r = str(range_).split("#")[-1] if range_ else "?"
        print(f"    - {pname} (range: {r})")


def main():
    print("=== Generating Premier League Ontology ===\n")

    g = create_ontology()

    print_ontology_stats(g)

    save_ontology(g)

    print("\nDone!")


if __name__ == "__main__":
    main()
