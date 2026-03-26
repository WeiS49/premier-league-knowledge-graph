"""
Premier League Wikipedia Crawler
Crawls team and player data from Wikipedia for building a knowledge graph
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from pathlib import Path

# base url for wikipedia
BASE_URL = "https://en.wikipedia.org"
PL_URL = "https://en.wikipedia.org/wiki/Premier_League"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) StudentProject/1.0'
}

# where we save stuff
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

# how many players per team to crawl (roughly)
MAX_PLAYERS_PER_TEAM = 5
DELAY = 1.5  # seconds between requests


def setup_dirs():
    """create output directories if they dont exist"""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"[+] Output dirs ready: {RAW_DIR}, {PROCESSED_DIR}")


def fetch_page(url):
    """fetch a wikipedia page and return soup object"""
    print(f"  fetching: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(DELAY)  # be polite
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"  [!] Error fetching {url}: {e}")
        return None


def save_raw_html(html, filename):
    """save raw html to data/raw/"""
    filepath = os.path.join(RAW_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)


def extract_infobox(soup):
    """Extract data from a Wikipedia infobox table
    Returns a dict of key-value pairs"""
    infobox = soup.find('table', class_='infobox')
    if not infobox:
        # try alternate class names
        infobox = soup.find('table', class_='infobox vcard')
    if not infobox:
        print("  [!] No infobox found on page")
        return {}

    data = {}
    rows = infobox.find_all('tr')
    for row in rows:
        header = row.find('th')
        value = row.find('td')
        if header and value:
            key = header.get_text(strip=True).lower().replace(' ', '_')
            val = value.get_text(strip=True)
            data[key] = val

    return data


def getTeamLinks(soup):
    """Get links to all current PL team pages from the main PL article.
    We look for the table with current teams."""

    team_links = []

    # look for tables that might contain team listings
    # Usually its in a table with "Club" header
    tables = soup.find_all('table', class_='wikitable')

    for table in tables:
        headers = table.find_all('th')
        header_texts = [h.get_text(strip=True).lower() for h in headers]

        # check if this looks like the clubs table
        if 'club' in header_texts or 'team' in header_texts:
            print(f"  [*] Found teams table with headers: {header_texts[:5]}")
            rows = table.find_all('tr')[1:]  # skip header row
            for row in rows:
                cells = row.find_all(['td', 'th'])
                for cell in cells:
                    link = cell.find('a')
                    if link and link.get('href', '').startswith('/wiki/'):
                        href = link['href']
                        # filter out non-team links
                        if ':' not in href and '#' not in href:
                            full_url = BASE_URL + href
                            team_name = link.get_text(strip=True)
                            if team_name and len(team_name) > 2:
                                team_links.append({
                                    'name': team_name,
                                    'url': full_url
                                })
                                break  # only first link per row
            break  # use first matching table

    # deduplicate
    seen = set()
    unique_links = []
    for t in team_links:
        if t['url'] not in seen:
            seen.add(t['url'])
            unique_links.append(t)

    return unique_links


def extract_team_data(soup, team_name, url):
    """extract structured data from a team's wikipedia page"""

    infobox_data = extract_infobox(soup)

    # map infobox fields to our schema
    team_data = {
        'name': team_name,
        'url': url,
        'full_name': infobox_data.get('full_name', infobox_data.get('fullname', '')),
        'ground': infobox_data.get('ground', ''),
        'league': infobox_data.get('league', 'Premier League'),
        'chairman': infobox_data.get('chairman', infobox_data.get('owner', '')),
        'manager': infobox_data.get('manager', infobox_data.get('head_coach', infobox_data.get('coach', ''))),
        'website': infobox_data.get('website', ''),
        'founded': infobox_data.get('founded', ''),
    }

    # also grab the first paragraph as description
    first_para = soup.find('div', class_='mw-parser-output')
    if first_para:
        p = first_para.find('p', class_=False)
        if p:
            team_data['description'] = p.get_text(strip=True)[:500]  # truncate

    return team_data


def get_player_links(soup, max_players=MAX_PLAYERS_PER_TEAM):
    """Get player links from team page.
    Looks for the squad section / current squad table"""

    player_links = []

    # Method 1: look for squad tables
    # TODO: this is fragile, different team pages have different formats
    squad_heading = None
    for heading in soup.find_all(['h2', 'h3']):
        text = heading.get_text(strip=True).lower()
        if 'squad' in text or 'player' in text or 'first.team' in text or 'current squad' in text:
            squad_heading = heading
            break

    if squad_heading:
        # get the next table or list after the heading
        next_elem = squad_heading.find_next_sibling()
        attempts = 0
        while next_elem and attempts < 10:
            if next_elem.name == 'table':
                # found a table, extract player links
                links = next_elem.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    if href.startswith('/wiki/') and ':' not in href:
                        name = link.get_text(strip=True)
                        # basic filter: player names are usually 2+ words
                        if name and len(name.split()) >= 2:
                            player_links.append({
                                'name': name,
                                'url': BASE_URL + href
                            })
                break
            elif next_elem.name in ['h2']:
                break  # went past the section
            next_elem = next_elem.find_next_sibling()
            attempts += 1

    # Method 2: if we didnt find enough, try looking for any table with player-like data
    if len(player_links) < 3:
        tables = soup.find_all('table', class_='wikitable')
        for table in tables:
            headerRow = table.find('tr')
            if headerRow:
                headers = [th.get_text(strip=True).lower() for th in headerRow.find_all('th')]
                if any(h in headers for h in ['name', 'player', 'no.', 'pos.']):
                    links = table.find_all('a')
                    for link in links:
                        href = link.get('href', '')
                        if href.startswith('/wiki/') and ':' not in href:
                            name = link.get_text(strip=True)
                            if name and len(name.split()) >= 2 and name not in [p['name'] for p in player_links]:
                                player_links.append({
                                    'name': name,
                                    'url': BASE_URL + href
                                })

    # deduplicate and limit
    seen = set()
    unique = []
    for p in player_links:
        if p['url'] not in seen:
            seen.add(p['url'])
            unique.append(p)

    return unique[:max_players]


def extract_player_data(soup, player_name, url):
    """extract player info from their wikipedia page"""

    infobox_data = extract_infobox(soup)

    player = {
        'name': player_name,
        'url': url,
        'birth_date': infobox_data.get('date_of_birth', infobox_data.get('born', '')),
        'birth_place': infobox_data.get('place_of_birth', ''),
        'position': infobox_data.get('playing_position', infobox_data.get('position', '')),
        'current_club': infobox_data.get('current_team', infobox_data.get('currentclub', '')),
        'nationality': infobox_data.get('nationality', ''),
    }

    # try to get youth clubs and senior clubs from career tables
    # TODO: parse the career tables more thoroughly
    youth_clubs = []
    senior_clubs = []

    # look for career info in infobox
    for key, val in infobox_data.items():
        if 'youth' in key:
            youth_clubs.append(val)
        elif 'senior' in key or 'club' in key.lower():
            if val and val != player.get('current_club'):
                senior_clubs.append(val)

    player['youth_clubs'] = youth_clubs
    player['senior_clubs'] = senior_clubs

    # get first paragraph
    content_div = soup.find('div', class_='mw-parser-output')
    if content_div:
        p = content_div.find('p', class_=False)
        if p:
            player['description'] = p.get_text(strip=True)[:500]

    return player


def crawl_teams():
    """Main function to crawl all PL teams"""
    print("\n" + "="*60)
    print("CRAWLING PREMIER LEAGUE TEAMS")
    print("="*60)

    # first get the main PL page
    html = fetch_page(PL_URL)
    if not html:
        print("[!!!] Failed to fetch Premier League page. Aborting.")
        return []

    save_raw_html(html, 'premier_league_main.html')
    soup = BeautifulSoup(html, 'html.parser')

    # get team links
    team_links = getTeamLinks(soup)
    print(f"\n[+] Found {len(team_links)} team links")

    if len(team_links) == 0:
        print("[!] No team links found... trying fallback")
        # hardcoded fallback for if the parsing fails
        # these are 2024-25 season teams
        team_links = [
            {'name': 'Arsenal F.C.', 'url': 'https://en.wikipedia.org/wiki/Arsenal_F.C.'},
            {'name': 'Aston Villa F.C.', 'url': 'https://en.wikipedia.org/wiki/Aston_Villa_F.C.'},
            {'name': 'AFC Bournemouth', 'url': 'https://en.wikipedia.org/wiki/AFC_Bournemouth'},
            {'name': 'Brentford F.C.', 'url': 'https://en.wikipedia.org/wiki/Brentford_F.C.'},
            {'name': 'Brighton & Hove Albion F.C.', 'url': 'https://en.wikipedia.org/wiki/Brighton_%26_Hove_Albion_F.C.'},
            {'name': 'Chelsea F.C.', 'url': 'https://en.wikipedia.org/wiki/Chelsea_F.C.'},
            {'name': 'Crystal Palace F.C.', 'url': 'https://en.wikipedia.org/wiki/Crystal_Palace_F.C.'},
            {'name': 'Everton F.C.', 'url': 'https://en.wikipedia.org/wiki/Everton_F.C.'},
            {'name': 'Fulham F.C.', 'url': 'https://en.wikipedia.org/wiki/Fulham_F.C.'},
            {'name': 'Ipswich Town F.C.', 'url': 'https://en.wikipedia.org/wiki/Ipswich_Town_F.C.'},
            {'name': 'Leicester City F.C.', 'url': 'https://en.wikipedia.org/wiki/Leicester_City_F.C.'},
            {'name': 'Liverpool F.C.', 'url': 'https://en.wikipedia.org/wiki/Liverpool_F.C.'},
            {'name': 'Manchester City F.C.', 'url': 'https://en.wikipedia.org/wiki/Manchester_City_F.C.'},
            {'name': 'Manchester United F.C.', 'url': 'https://en.wikipedia.org/wiki/Manchester_United_F.C.'},
            {'name': 'Newcastle United F.C.', 'url': 'https://en.wikipedia.org/wiki/Newcastle_United_F.C.'},
            {'name': 'Nottingham Forest F.C.', 'url': 'https://en.wikipedia.org/wiki/Nottingham_Forest_F.C.'},
            {'name': 'Southampton F.C.', 'url': 'https://en.wikipedia.org/wiki/Southampton_F.C.'},
            {'name': 'Tottenham Hotspur F.C.', 'url': 'https://en.wikipedia.org/wiki/Tottenham_Hotspur_F.C.'},
            {'name': 'West Ham United F.C.', 'url': 'https://en.wikipedia.org/wiki/West_Ham_United_F.C.'},
            {'name': 'Wolverhampton Wanderers F.C.', 'url': 'https://en.wikipedia.org/wiki/Wolverhampton_Wanderers_F.C.'},
        ]
        print(f"  Using {len(team_links)} hardcoded team links")

    # crawl each team
    teams = []
    for i, team in enumerate(team_links[:20]):  # limit to 20
        print(f"\n--- Team {i+1}/{min(len(team_links), 20)}: {team['name']} ---")
        html = fetch_page(team['url'])
        if html is None:
            continue

        # save raw html
        safe_name = re.sub(r'[^\w]', '_', team['name'])
        save_raw_html(html, f'team_{safe_name}.html')

        soup = BeautifulSoup(html, 'html.parser')
        team_data = extract_team_data(soup, team['name'], team['url'])
        teams.append(team_data)
        print(f"  -> Extracted: {team_data.get('full_name', team['name'])}")

    return teams


def crawl_players(teams):
    """Crawl player pages for each team"""
    print("\n" + "="*60)
    print("CRAWLING PLAYER DATA")
    print("="*60)

    all_players = []
    crawled_urls = set()  # avoid crawling same player twice

    for team in teams:
        print(f"\n--- Getting players for: {team['name']} ---")

        # re-fetch team page to get player links
        # (yeah this is inefficient, we could cache but whatever)
        # TODO: cache the team page soup instead of re-fetching
        html = fetch_page(team['url'])
        if not html:
            continue

        soup = BeautifulSoup(html, 'html.parser')
        player_links = get_player_links(soup)
        print(f"  Found {len(player_links)} player links")

        for plink in player_links:
            if plink['url'] in crawled_urls:
                print(f"  [skip] Already crawled: {plink['name']}")
                continue

            crawled_urls.add(plink['url'])
            html = fetch_page(plink['url'])
            if html is None:
                continue

            safe_name = re.sub(r'[^\w]', '_', plink['name'])
            save_raw_html(html, f'player_{safe_name}.html')

            try:
                soup = BeautifulSoup(html, 'html.parser')
                player_data = extract_player_data(soup, plink['name'], plink['url'])
                player_data['team'] = team['name']
                all_players.append(player_data)
                print(f"  -> {player_data['name']} ({player_data.get('position', 'unknown')})")
            except:
                print(f"  [!] Failed to parse player: {plink['name']}")
                continue

    return all_players


def save_data(teams, players):
    """save structured data to json files"""
    teams_path = os.path.join(PROCESSED_DIR, 'teams.json')
    players_path = os.path.join(PROCESSED_DIR, 'players.json')

    with open(teams_path, 'w', encoding='utf-8') as f:
        json.dump(teams, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved {len(teams)} teams to {teams_path}")

    with open(players_path, 'w', encoding='utf-8') as f:
        json.dump(players, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(players)} players to {players_path}")


def main():
    print("Premier League Wikipedia Crawler")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Delay between requests: {DELAY}s")

    setup_dirs()

    # crawl teams
    teams = crawl_teams()
    print(f"\n[RESULT] Crawled {len(teams)} teams")

    # crawl players
    players = crawl_players(teams)
    print(f"\n[RESULT] Crawled {len(players)} players")

    # save everything
    save_data(teams, players)

    print(f"\nDone! Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total: {len(teams)} teams, {len(players)} players")


if __name__ == '__main__':
    main()
