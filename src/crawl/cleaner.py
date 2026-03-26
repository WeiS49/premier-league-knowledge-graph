"""
Data Cleaning Script
Cleans the raw JSON data from the crawler and prepares it for the knowledge graph
"""

import json, os, re
from datetime import datetime
from pathlib import Path

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')


def load_json(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    print(f"Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} records")
    return data


def cleanText(text):
    """Remove wikipedia citation markers and clean up text.
    Handles things like [1], [2], [citation needed], [a], etc."""
    if not isinstance(text, str):
        return text

    # remove citation brackets like [1], [23], [a], [citation needed]
    text = re.sub(r'\[(?:\d+|[a-z]|citation needed|clarification needed|when\?|who\?)\]', '', text, flags=re.IGNORECASE)

    # remove leftover bracket references we might have missed
    text = re.sub(r'\[\d+\]', '', text)

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # remove leading/trailing parentheses that are empty
    text = re.sub(r'\(\s*\)', '', text)

    return text


def clean_dict(d):
    """recursively clean all string values in a dict"""
    cleaned = {}
    for key, val in d.items():
        if isinstance(val, str):
            cleaned[key] = cleanText(val)
        elif isinstance(val, list):
            cleaned[key] = [cleanText(v) if isinstance(v, str) else v for v in val]
        elif isinstance(val, dict):
            cleaned[key] = clean_dict(val)
        else:
            cleaned[key] = val
    return cleaned


def normalize_date(date_str):
    """Try to parse a date string and return ISO format.
    Wikipedia dates come in many formats so we try a few."""
    if not date_str or not isinstance(date_str, str):
        return date_str

    # clean first
    date_str = cleanText(date_str)

    # remove age in parentheses like "(age 25)"
    date_str = re.sub(r'\(age\s*\d+\)', '', date_str, flags=re.IGNORECASE).strip()
    # remove "aged XX"
    date_str = re.sub(r'aged\s*\d+', '', date_str, flags=re.IGNORECASE).strip()

    # try different date formats
    formats_to_try = [
        '%d %B %Y',      # 15 March 1990
        '%B %d, %Y',     # March 15, 1990
        '%Y-%m-%d',      # 1990-03-15
        '%d %b %Y',      # 15 Mar 1990
        '%d/%m/%Y',      # 15/03/1990
        '%Y',            # just a year like 1886
        '%d %B%Y',       # sometimes space is missing: "15 March1990"
    ]

    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if fmt == '%Y':
                return date_str.strip()  # just return the year
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue

    # if nothing worked, try to at least extract a year
    year_match = re.search(r'(\d{4})', date_str)
    if year_match:
        return date_str  # return as-is but at least its cleaned

    return date_str


def deduplicate(items, key_field='name'):
    """Remove duplicate entries based on a key field.
    Keeps the first occurrence."""
    seen = set()
    unique = []
    dupes = 0
    for item in items:
        key = item.get(key_field, '').lower().strip()
        if key and key in seen:
            dupes += 1
            continue
        seen.add(key)
        unique.append(item)

    if dupes > 0:
        print(f"  Removed {dupes} duplicates (by {key_field})")
    return unique


def clean_teams(teams):
    """Clean team data"""
    print("\n--- Cleaning Teams ---")
    cleaned = []

    stats = {'total': len(teams), 'cleaned_fields': 0, 'dates_normalized': 0}

    for team in teams:
        t = clean_dict(team)

        # normalize founded date
        if t.get('founded'):
            original = t['founded']
            t['founded'] = normalize_date(t['founded'])
            if t['founded'] != original:
                stats['dates_normalized'] += 1

        # make sure league field is set
        if not t.get('league'):
            t['league'] = 'Premier League'

        # clean up website urls
        if t.get('website'):
            website = t['website']
            if not website.startswith('http'):
                t['website'] = 'https://' + website

        cleaned.append(t)
        stats['cleaned_fields'] += 1

    # deduplicate
    cleaned = deduplicate(cleaned, 'name')

    print(f"  Total teams: {stats['total']}")
    print(f"  Dates normalized: {stats['dates_normalized']}")
    print(f"  Final count after dedup: {len(cleaned)}")

    return cleaned


def clean_players(players):
    """Clean player data"""
    print("\n--- Cleaning Players ---")
    cleaned = []

    stats = {'total': len(players), 'dates_normalized': 0, 'positions_cleaned': 0}

    for player in players:
        p = clean_dict(player)

        # normalize birth date
        if p.get('birth_date'):
            original = p['birth_date']
            p['birth_date'] = normalize_date(p['birth_date'])
            if p['birth_date'] != original:
                stats['dates_normalized'] += 1

        # clean position field - sometimes has extra info
        if p.get('position'):
            pos = p['position']
            # standardize common positions
            pos_map = {
                'goalkeeper': 'Goalkeeper',
                'gk': 'Goalkeeper',
                'defender': 'Defender',
                'centre-back': 'Centre-back',
                'center-back': 'Centre-back',
                'left-back': 'Left-back',
                'right-back': 'Right-back',
                'midfielder': 'Midfielder',
                'central midfielder': 'Central Midfielder',
                'attacking midfielder': 'Attacking Midfielder',
                'defensive midfielder': 'Defensive Midfielder',
                'forward': 'Forward',
                'striker': 'Striker',
                'winger': 'Winger',
                'left winger': 'Left Winger',
                'right winger': 'Right Winger',
            }
            pos_lower = pos.lower().strip()
            if pos_lower in pos_map:
                p['position'] = pos_map[pos_lower]
                stats['positions_cleaned'] += 1

        # remove empty lists
        if p.get('youth_clubs') == []:
            del p['youth_clubs']
        if p.get('senior_clubs') == []:
            del p['senior_clubs']

        cleaned.append(p)

    # deduplicate by name
    cleaned = deduplicate(cleaned, 'name')

    print(f"  Total players: {stats['total']}")
    print(f"  Dates normalized: {stats['dates_normalized']}")
    print(f"  Positions standardized: {stats['positions_cleaned']}")
    print(f"  Final count after dedup: {len(cleaned)}")

    return cleaned


def buildCorpus(teams, players):
    """Build a text corpus from all the data for NER processing.
    Combines descriptions and key text fields into a single text file."""
    print("\n--- Building Text Corpus ---")

    corpus_lines = []

    # add team descriptions
    for team in teams:
        if team.get('description'):
            corpus_lines.append(team['description'])
        # also add a structured line
        parts = []
        if team.get('full_name'):
            parts.append(team['full_name'])
        if team.get('ground'):
            parts.append(f"plays at {team['ground']}")
        if team.get('manager'):
            parts.append(f"managed by {team['manager']}")
        if team.get('founded'):
            parts.append(f"founded {team['founded']}")
        if parts:
            corpus_lines.append('. '.join(parts) + '.')

    # add player info
    for player in players:
        if player.get('description'):
            corpus_lines.append(player['description'])
        # structured line
        parts = [player.get('name', 'Unknown')]
        if player.get('position'):
            parts.append(f"plays as {player['position']}")
        if player.get('current_club'):
            parts.append(f"for {player['current_club']}")
        if player.get('nationality'):
            parts.append(f"nationality {player['nationality']}")
        if player.get('birth_place'):
            parts.append(f"born in {player['birth_place']}")
        corpus_lines.append('. '.join(parts) + '.')

    corpus_text = '\n\n'.join(corpus_lines)
    print(f"  Corpus size: {len(corpus_lines)} entries, {len(corpus_text)} characters")

    return corpus_text


def save_cleaned(teams, players, corpus):
    """save all cleaned data"""
    teams_path = os.path.join(PROCESSED_DIR, 'teams_clean.json')
    players_path = os.path.join(PROCESSED_DIR, 'players_clean.json')
    corpus_path = os.path.join(PROCESSED_DIR, 'corpus.txt')

    with open(teams_path, 'w', encoding='utf-8') as f:
        json.dump(teams, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved cleaned teams to {teams_path}")

    with open(players_path, 'w', encoding='utf-8') as f:
        json.dump(players, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved cleaned players to {players_path}")

    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(corpus)
    print(f"[+] Saved corpus ({len(corpus)} chars) to {corpus_path}")


def main():
    print("="*60)
    print("DATA CLEANING PIPELINE")
    print("="*60)

    # load raw data
    try:
        teams_raw = load_json('teams.json')
        players_raw = load_json('players.json')
    except FileNotFoundError as e:
        print(f"[!!!] Could not find input files: {e}")
        print("Make sure you run crawler.py first!")
        return

    # clean
    teams_clean = clean_teams(teams_raw)
    players_clean = clean_players(players_raw)

    # build corpus
    corpus = buildCorpus(teams_clean, players_clean)

    # save
    save_cleaned(teams_clean, players_clean, corpus)

    # print summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Teams:   {len(teams_raw)} raw -> {len(teams_clean)} clean")
    print(f"Players: {len(players_raw)} raw -> {len(players_clean)} clean")
    print(f"Corpus:  {len(corpus)} characters")
    print("Done!")


if __name__ == '__main__':
    main()
