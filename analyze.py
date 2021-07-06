from typing import List, Tuple, Dict
from collections import Counter

import numpy
import pandas as pd
import seaborn as sns
import stanza as stanza
import re
from click import secho
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from pandas.core.strings import StringMethods
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from datasets import load_dataset

from model_evaluater import evaluate

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

        sns.barplot(x=y, y=HebrewSongs.invert_words(x))
        plt.show()

    def get_song_length_from_years(self):
        self.data['lyrics_len'] = [len(lyric.split(' ')) for lyric in self.data['lyrics']]
        self.data['decade'] = [int(int(year) / 10) * 10 if type(year) != float and year.isdigit() else 0 for year in
                               self.data["year"]]
        print(self.data[["lyrics_len", "decade"]].groupby("decade").mean())

    def get_ngram_most_common(self, n: int, df: DataFrame = DataFrame()):
        all_lyrics: Series = self.data["lyrics"] if df.empty else df["lyrics"]

        vec = CountVectorizer(ngram_range=(3, 4)).fit(all_lyrics)
        bag_of_words = vec.transform(all_lyrics)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        top_n_bigrams: List[Tuple[str]] = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
        # without_stop_words = [bigram for bigram in bigrams 
        #                   if (bigram[0].split()[0] not in stop_words 
        #                       and bigram[0].split()[0] not in stop_words)][:20]
        x, y = map(list, zip(*top_n_bigrams))
        sns.barplot(x=y, y=HebrewSongs.invert_words(x))
        plt.show()

    def learn(self):
        df = DataFrame()
        X = self.data["lyrics"]
        y = self.data["hit"]
        vectorizer = CountVectorizer()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        mnb_prediction = mnb.predict(X_test)
        y_test = numpy.array(y_test)
        evaluate(mnb_prediction, y_test)
        return X_train, X_test, y_train, y_test, vectorizer

    @staticmethod
    def get_stop_words():
        with open('stopwords.txt', 'r', encoding='utf8') as f:
            return f.read().split()

    @staticmethod
    def invert_words(words):
        return [w[::-1] for w in words]


def subject_of_sentence():
    stanza.download('he')
    snlp = stanza.Pipeline(lang="he")
    for sentence in ['המכונית נסעה בשכונה']:
        a = snlp(sentence)
        for word in a.sentences[0].words:
            if word.deprel == "nsubj":
                print('the subject')
                print(word)
            if word.feats is not None and 'Gender' in word.feats:
                matcher = re.match(r'Gender=(\S+?)\|', word.feats).group(1)
                print(matcher)
                print(word.feats)
        print('------------')


def write_test_and_train_csv():
    for data_type in ["test", 'train']:
        dataset = load_dataset("hebrew_sentiment")
        labels = []
        texts = []
        MAX_LEN_ENCODE = 120
        for data in dataset[data_type]:
            text = data["text"].replace(',', '')
            if len(text) < MAX_LEN_ENCODE and int(data["label"]) !=2:
                labels.append(data["label"])
                texts.append(deEmojify(text))

        df = pd.DataFrame({'label': labels, 'text': texts}, columns=['label', 'text'])
        df.to_csv(f'{data_type}_sentiment.csv')


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


if __name__ == '__main__':
    # model = HebrewSongs('data.tsv')
    # model.get_song_length_from_years()
    write_test_and_train_csv()
