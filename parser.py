import re
import math
from typing import List, Dict
from urllib.parse import urlparse, parse_qs

import pandas
import requests
from tqdm import tqdm
from click import secho
from pandas import DataFrame
from bs4 import BeautifulSoup

BASE_URL = 'https://shironet.mako.co.il'
ABC_INDECIES = list(range(22))
ABC = 'abcdefghijklmnopqrstuvwxysABCDEFGHIJKLMNOPQRSTUVWXYZ'


def read_url(url: str) -> str:
    try:
        response = requests.get(url)
        return response.content.decode('utf-8')
    except:
        return ""


def get_artists(load: bool = True, save: bool = False) -> List[Dict[str, str]]:
    if load:
        df = pandas.read_csv('artists.tsv', sep='\t')
        return df.to_dict('records')

    artists = []
    for letter in tqdm(ABC_INDECIES):
        if letter < 14: continue
        try:
            url = f'{BASE_URL}/html/indexes/performers/heb_{letter}_popular.html'
            content = read_url(url)
            soup = BeautifulSoup(content, 'html.parser')
            search_results = soup.find('table', attrs={'class': 'search_results'})
            artists.extend([{"name": a.text.strip(), "id": parse_qs(urlparse(a["href"]).query)["prfid"][0]} for td in search_results.find_all('td') for a in td.find_all('a', href=True)])
        except:
            secho(f'[WARNING] failed to parse url {url}')
    if save:
        df = DataFrame(artists)
        df.to_csv('artists.tsv', sep="\t", index=False, encoding='utf-8')
    return artists


def get_number_of_pages(artist_id: int) -> int:
    url = f'https://shironet.mako.co.il/artist?type=works&lang=1&prfid={artist_id}'
    content = read_url(url)
    soup = BeautifulSoup(content, 'html.parser')
    try:
        # we first locate some known place -> artist_player_songlist.
        search = soup.find_all('a', attrs={'class': 'artist_player_songlist'})[0].find_parent().find_parent().find_parent().find_parent().find_parent().find_parent()

        # there is no unique identifier for this tag, so we do it ugly.
        number_of_songs = search.find_all('table')[4].find_all('span', attrs={'class': 'artist_normal_txt'})[1].find_all('strong')[1].text.strip()

        # there are a most 30 songs in a page
        return math.ceil(float(number_of_songs) / 30)
    except:
        return 0


def get_artist_songs(load: bool = True, save: bool = False) -> List[Dict[str, str]]:
    if load:
        df = pandas.read_csv('songs.tsv', sep='\t')
        return df.to_dict('records')

    artists = get_artists()
    songs = []
    for artist in tqdm(artists):
        number_of_pages = get_number_of_pages(artist["id"])
        for page in range(number_of_pages):
            try:
                url = f'https://shironet.mako.co.il/artist?type=works&lang=1&prfid={artist["id"]}&page={page}'
                content = read_url(url)
                soup = BeautifulSoup(content, 'html.parser')
                songs.extend([
                    {"song": song.text.strip(), "link": f'{BASE_URL}{song["href"]}', "artist_name": artist["name"], "artist_id": artist["id"]} 
                    for song in soup.find_all('a', attrs={'class': 'artist_player_songlist'}, href=True) 
                    if not re.search('[a-zA-Z]', song.text.strip())
                ])
            except:
                secho(f'[WARNING] failed to parse url {url}')
    if save:
        df = DataFrame(songs)
        df.to_csv('songs.tsv', sep="\t", index=False, encoding='utf-8')
    return songs




