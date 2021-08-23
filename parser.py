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
ABC = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
GOOGLE_URL = 'https://www.google.com'


def read_url(url: str, decode: bool = True) -> str:
    try:
        response = requests.get(url)
        if decode:
            return response.content.decode('utf-8')
        else:
            return response.content
    except:
        return ""


def get_artists(load: bool = True, save: bool = False) -> List[Dict[str, str]]:
    if load:
        df = pandas.read_csv('artists.tsv', sep='\t')
        return df.to_dict('records')

    artists = []
    for letter in tqdm(ABC_INDECIES):
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
    url = f'{BASE_URL}/artist?type=works&lang=1&prfid={artist_id}'
    content = read_url(url)
    soup = BeautifulSoup(content, 'html.parser')
    try:
        # we first locate some known place -> artist_player_songlist.
        search = soup.find_all('a', attrs={'class': 'artist_player_songlist'})[0].find_parent().find_parent().find_parent().find_parent().find_parent().find_parent()

        # there is no unique identifier for this tag, so we do it ugly.
        number_of_songs = search.find_all('table')[4].find_all('strong')[1].text.strip()
        
        # there are a most 30 songs in a page
        return math.ceil(float(number_of_songs) / 30)
    except:
        return -1


def get_artist_songs(load: bool = True, save: bool = False) -> List[Dict[str, str]]:
    if load:
        df = pandas.read_csv('songs2.tsv', sep='\t')
        return df.to_dict('records')

    artists = get_artists()
    songs = []
    for artist in tqdm(artists):

        number_of_pages = get_number_of_pages(artist["id"])
        if number_of_pages == -1:
            print(artist['name'])
            continue
        for page in range(1, number_of_pages + 1):
            try:
                url = f'{BASE_URL}/artist?type=works&lang=1&prfid={artist["id"]}&page={page}'
                content = read_url(url)
                soup = BeautifulSoup(content, 'html.parser')
                songs.extend([
                    {"song": song.text.strip(), "link": f'{BASE_URL}{song["href"]}', "artist_name": artist["name"], "artist_id": artist["id"]} 
                    for song in soup.find_all('a', attrs={'class': 'artist_player_songlist'}, href=True) 
                    if not re.search('[a-zA-Z]', song.text.strip())
                ])
            except:
                secho(f'[WARNING] failed to parse url {url}', fg="red")
    if save:
        df = DataFrame(songs)
        df.to_csv('songs3.tsv', sep="\t", index=False, encoding='utf-8')
    return songs


def get_song_lyrics(url: str):
    content = read_url(url)
    soup = BeautifulSoup(content, 'html.parser')
    lyrics = soup.find('span', attrs={'class': 'artist_lyrics_text'}).text
    return lyrics


def get_song_year(artist: str, song: str) -> int:
    url = f'{GOOGLE_URL}/search?q=תאריך+הפצה+{song.replace(" ", "+")}+{artist.replace(" ", "+")}'
    try:
        content = read_url(url, decode=False)
        soup = BeautifulSoup(content, 'html.parser')
        return soup.find("span", string="תאריך הפצה").find_parent().find_parent().find_all('div')[-1].text.strip()
    except:
        secho(f"[WARNING] could not parse {url}", fg="red")
        return -1


def build_tsv(output: str):
    songs = get_artist_songs(load=False)
    for song in tqdm(songs):
        try:
            song["lyrics"] = get_song_lyrics(song["link"])
        except:
            secho(f"couldnt parse {song['song']}", fg="red", bold=True)
            song["lyrics"] = "להשלים להשלים להשלים"
    df = DataFrame(songs)
    df.to_csv(output, sep="\t", index=False, encoding='utf-8')


def add_year_to_data(input: str, output: str):
    df = pandas.read_csv(input, sep='\t')
    dict_to_edit = df.to_dict('records')
    i = 0
    for row in tqdm(dict_to_edit):
        row["year"] = get_song_year(row["song"], row["artist_name"])
        i += 1
        if i % 50 == 0:
            new_df = DataFrame(dict_to_edit)
            new_df.to_csv(output, sep="\t", index=False, encoding='utf-8')
    if len(dict_to_edit) < 50:
        new_df = DataFrame(dict_to_edit)
        new_df.to_csv(output, sep="\t", index=False, encoding='utf-8')

def add_hit_to_data(input: str, output: str):
    df = pandas.read_csv(input, sep='\t')
    dict_to_edit = df.to_dict('records')
    for row in tqdm(dict_to_edit):
        row["hit"] = 0
        try:
            row["year"] = str(row["year"])[:-2]
        except:
            row["year"] = -1
    new_df = DataFrame(dict_to_edit)
    new_df.to_csv(output, sep="\t", index=False, encoding='utf-8')

def update_hit():
    hits = pandas.read_csv("bla.csv")
    data = pandas.read_csv("data_with_years2.tsv", sep='\t')
    data_to_edit = data.to_dict('records')
    count = 0
    for i, hit in hits.iterrows():
        for row in data_to_edit:
            if hit["song"] == row["song"] and hit["link"] == row["link"] and hit["artist_name"] == row["artist_name"] and str(hit["artist_id"]) == str(row["artist_id"])[:-2]:
                row["hit"] = hit["hit"]
                row["year"] = hit["year"]
                row["artist_id"] = hit["artist_id"]
                count += 1
    print(count)
    new_df = DataFrame(data_to_edit)
    new_df.to_csv("data_with_years.tsv", sep="\t", index=False, encoding='utf-8')


def drop_non_year():
    df = pandas.read_csv("data.tsv", sep='\t')
    df = df[df['year'] != "-1"]
    df.to_csv("data2.tsv", sep="\t", index=False, encoding='utf-8')


def add_artists_gender():
    artists_df = pandas.read_csv("artists.csv")
    artists_dict = {}
    for i, row in artists_df.iterrows():
        artists_dict[row["artist_name"]] = row["gender"]

    df = pandas.read_csv("data.tsv", sep='\t')
    for i, row in df.iterrows():
        df.loc[i, "gender"] = artists_dict[row["artist_name"]]

    df.to_csv("data.tsv", sep="\t", index=False, encoding='utf-8')


if __name__ == '__main__':
    ######################################
    ## NO NEED TO USE THIS FILE ANYMORE ##
    ######################################
    # pass

    add_artists_gender()

    