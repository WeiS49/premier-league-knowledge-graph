"""
SWRL reasoning with OWLReady2
- Part 1: family ontology old person rule
- Part 2: football knowledge base with player/team/league inference
"""

import os
from pathlib import Path
from owlready2 import *


# TODO: maybe make this configurable later
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "reason"


def run_family_swrl():
    """Load family_lab.owl, add SWRL rule for OldPerson, run reasoner"""

    print("Loading family ontology...")
    owl_path = PROJECT_ROOT / "docs" / "TD3" / "TP3" / "family_lab.owl"
    print(f"  path: {owl_path}")

    if not owl_path.exists():
        print(f"ERROR: file not found: {owl_path}")
        return

    # load the ontology
    onto = get_ontology(str(owl_path)).load()
    print(f"  loaded! base IRI = {onto.base_iri}")

    # classes live in the unnamed namespace
    ns = get_ontology("http://www.owl-ontologies.com/unnamed.owl#")

    print(f"  classes: {[c.name for c in onto.classes()]}")
    print(f"  individuals: {list(onto.individuals())}")
    print(f"  data properties: {[p.name for p in onto.data_properties()]}")

    Person = ns.Person
    age_prop = ns.age

    with onto:
        # create OldPerson as subclass of Person
        if not ns["OldPerson"]:
            print("  creating OldPerson class...")
            class OldPerson(Person):
                pass
        else:
            OldPerson = ns["OldPerson"]

        # add test individuals if none exist
        if not list(onto.individuals()):
            print("  No individuals found, adding test persons...")
            test_data = [("Jean", 70), ("Marie_T", 45), ("Pierre_B", 80), ("Sophie_P", 30)]
            for pname, page in test_data:
                p = Person(pname, namespace=onto)
                age_prop[p].append(page)
            print(f"  added: Jean(70), Marie_T(45), Pierre_B(80), Sophie_P(30)")

        # debug: check ages
        for ind in onto.individuals():
            ages = age_prop[ind]
            if ages:
                print(f"  DEBUG: {ind.name} has age = {ages}")

        # Add SWRL rule using programmatic API
        # Person(?p) ^ age(?p, ?a) ^ greaterThan(?a, 60) -> OldPerson(?p)
        print(f"\n  Adding SWRL rule: Person(?p) ^ age(?p, ?a) ^ greaterThan(?a, 60) -> OldPerson(?p)")
        rule = Imp()
        rule.set_as_rule(
            """Person(?p) ^ age(?p, ?a) ^ greaterThan(?a, 60) -> OldPerson(?p)""",
            namespaces=[onto, ns]
        )
        print(f"  Rule added successfully")

    # Run the reasoner
    print("\n  Running reasoner...")
    try:
        sync_reasoner_pellet(infer_property_values=True,
                             infer_data_property_values=True)
        print("  Pellet reasoner finished!")
    except Exception as e:
        print(f"  Pellet failed ({e}), trying HermiT...")
        try:
            sync_reasoner(infer_property_values=True)
            print("  HermiT reasoner finished!")
        except Exception as e2:
            print(f"  ERROR: Both reasoners failed: {e2}")
            print("  Falling back to manual classification...")
            # manual fallback - just classify based on age directly
            for ind in onto.individuals():
                ages = age_prop[ind]
                if ages and ages[0] > 60:
                    ind.is_a.append(OldPerson)
            print("  Manual classification done")

    # Check results
    OldPerson = onto["OldPerson"] or ns["OldPerson"]
    if OldPerson is None:
        print("  Could not find OldPerson class after reasoning")
        return
    old_persons = list(OldPerson.instances())
    print(f"\n  === OldPerson instances (age > 60) ===")
    if old_persons:
        for p in old_persons:
            ages = age_prop[p]
            print(f"    - {p.name}: age = {ages}")
    else:
        print("    (none found)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = OUTPUT_DIR / "family_inferred.owl"
    onto.save(file=str(output_path), format="rdfxml")
    print(f"\n  Saved inferred ontology to: {output_path}")


def run_football_swrl():
    """Create football ontology with SWRL rule for playsInLeague inference"""

    print("Creating football ontology from scratch...")

    onto = get_ontology("http://example.org/football.owl")

    with onto:
        class Person(Thing):
            pass
        class Player(Person):
            pass
        class Team(Thing):
            pass
        class League(Thing):
            pass

        class playsFor(ObjectProperty):
            domain = [Player]
            range = [Team]
        class competesIn(ObjectProperty):
            domain = [Team]
            range = [League]
        class playsInLeague(ObjectProperty):
            domain = [Player]
            range = [League]

        # individuals
        print("  Adding individuals...")
        ligue1 = League("Ligue1")
        premier_league = League("PremierLeague")

        psg = Team("PSG")
        psg.competesIn = [ligue1]
        marseille = Team("OlympiqueMarseille")
        marseille.competesIn = [ligue1]
        arsenal = Team("Arsenal")
        arsenal.competesIn = [premier_league]

        mbappe = Player("Mbappe")
        mbappe.playsFor = [psg]
        payet = Player("Payet")
        payet.playsFor = [marseille]
        saka = Player("Saka")
        saka.playsFor = [arsenal]
        havertz = Player("Havertz")
        havertz.playsFor = [arsenal]

        print(f"  Players: Mbappe(PSG), Payet(OM), Saka(Arsenal), Havertz(Arsenal)")
        print(f"  Teams: PSG->Ligue1, OM->Ligue1, Arsenal->PremierLeague")

        # SWRL rule
        rule_str = "Player(?p) ^ playsFor(?p, ?t) ^ competesIn(?t, ?l) -> playsInLeague(?p, ?l)"
        print(f"\n  Adding SWRL rule: {rule_str}")
        rule = Imp()
        rule.set_as_rule(rule_str)

    print("\n  Before reasoning:")
    for player in Player.instances():
        print(f"    {player.name}.playsInLeague = {player.playsInLeague}")

    print("\n  Running reasoner...")
    try:
        sync_reasoner_pellet(infer_property_values=True,
                             infer_data_property_values=True)
        print("  Pellet reasoner finished!")
    except Exception as e:
        print(f"  Pellet failed ({e}), trying HermiT...")
        try:
            sync_reasoner(infer_property_values=True)
            print("  HermiT reasoner finished!")
        except Exception as e2:
            print(f"  Reasoners failed: {e2}")
            print("  Falling back to manual inference...")
            for player in Player.instances():
                for team in player.playsFor:
                    for league in team.competesIn:
                        if league not in player.playsInLeague:
                            player.playsInLeague.append(league)
            print("  Manual inference done")

    # results
    print(f"\n  === Inferred playsInLeague relationships ===")
    for player in Player.instances():
        for league in player.playsInLeague:
            print(f"    {player.name} playsInLeague {league.name}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = OUTPUT_DIR / "football_inferred.owl"
    onto.save(file=str(output_path), format="rdfxml")
    print(f"\n  Saved football ontology to: {output_path}")


def main():
    print("=" * 60)
    print("  SWRL Rules with OWLReady2")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("PART 1: Family Ontology - OldPerson Rule")
    print("-" * 60 + "\n")
    run_family_swrl()

    print("\n" + "-" * 60)
    print("PART 2: Football KB - playsInLeague Inference")
    print("-" * 60 + "\n")
    run_football_swrl()

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
