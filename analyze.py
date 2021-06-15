from typing import List, Tuple, Dict
from collections import Counter

import pandas
import seaborn as sns
from click import secho
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from pandas.core.strings import StringMethods

DECADES = [1970, 1980, 1990, 2000, 2010, 2020]

class HebrewSongs:
    def __init__(self, path: str):
         self.data: DataFrame = pandas.read_csv(path, sep='\t')

    def get_decade(self, decade: int) -> DataFrame:
        if decade not in DECADES:
            secho(f"[ERROR] decade ({decade}) should be one of the following:", fg="red", bold=True)
            secho(f"        {DECADES}", fg="red")
            exit(1)
        return self.data.loc[(self.data['year'] >= decade) & (self.data['year'] <= (decade + 9))]

    def get_year(self, year: int) -> DataFrame:
        return self.data.loc[self.data['year'] == year]
    
    def get_lyrics(self, df: DataFrame = DataFrame()):
        all_lyrics: Series = self.data["lyrics"] if df.empty else df["lyrics"]
        all_lyrics_str: StringMethods = all_lyrics.str
        words: Series = all_lyrics_str.split()
        words: List[List[str]] = words.values.tolist()
        return [word for i in words for word in i]

    def get_most_common(self, n: int, decade: int = None):
        corpus: List[str] = self.get_lyrics(self.get_decade(decade) if decade else DataFrame())
        counter = Counter(corpus)
        most = counter.most_common()
        x, y = [], []
        for word, count in most[:n]:
            x.append(word)
            y.append(count)
        sns.barplot(x=y,y=HebrewSongs.invert_words(x))
        plt.show()

    @staticmethod
    def invert_words(words):
        return [w[::-1] for w in words]


if __name__ == '__main__':
    model = HebrewSongs('data_copy_with_years.tsv')
    model.get_most_common(n=40, decade=1980)
    

