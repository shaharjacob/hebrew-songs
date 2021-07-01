from typing import List, Tuple, Dict
from collections import Counter

import pandas as pd
import seaborn as sns
from click import secho
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from pandas.core.strings import StringMethods
from sklearn.feature_extraction.text import CountVectorizer

DECADES = [1970, 1980, 1990, 2000, 2010, 2020]

class HebrewSongs:
    def __init__(self, path: str):
         self.data: DataFrame = pd.read_csv(path, sep='\t')
         self.stop_words: List[str] = HebrewSongs.get_stop_words()

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

    def get_most_common(self, n: int, decade: int = None, use_stopwords: bool = True):
        corpus: List[str] = self.get_lyrics(self.get_decade(decade) if decade else DataFrame())
        counter = Counter(corpus)
        most = counter.most_common()
        x, y = [], []
        for word, count in most:
            if use_stopwords:
                if word not in self.stop_words:
                    x.append(word)
                    y.append(count)
            else:
                x.append(word)
                y.append(count)

            if len(x) == n:
              break

        sns.barplot(x=y,y=HebrewSongs.invert_words(x))
        plt.show()

    def get_ngram_most_common(self, n: int, df: DataFrame = DataFrame()):
        all_lyrics: Series = self.data["lyrics"] if df.empty else df["lyrics"]
        vec = CountVectorizer(ngram_range=(3, 4)).fit(all_lyrics)
        bag_of_words = vec.transform(all_lyrics)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                    for word, idx in vec.vocabulary_.items()]
        top_n_bigrams: List[Tuple[str]] = sorted(words_freq, key = lambda x: x[1], reverse=True)[:n]
        # without_stop_words = [bigram for bigram in bigrams 
        #                   if (bigram[0].split()[0] not in stop_words 
        #                       and bigram[0].split()[0] not in stop_words)][:20]
        x, y = map(list, zip(*top_n_bigrams))
        sns.barplot(x=y,y=HebrewSongs.invert_words(x))
        plt.show()

    @staticmethod
    def get_stop_words():
        with open('stopwords.txt','r', encoding='utf8') as f:
            return f.read().split()

    @staticmethod
    def invert_words(words):
        return [w[::-1] for w in words]


if __name__ == '__main__':
    model = HebrewSongs('data.tsv')
    # model.get_most_common(n=40, use_stopwords=True)
    model.get_ngram_most_common(n=40)

