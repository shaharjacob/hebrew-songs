import re
from typing import List, Tuple, Set
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from click import secho
import stanza as stanza
import matplotlib.pyplot as plt
from datasets import load_dataset
from pandas import DataFrame, Series
from pandas.core.strings import StringMethods
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text

from model_evaluater import evaluate

DECADES = [1970, 1980, 1990, 2000, 2010, 2020]
EPOCHS = 3
ISRAEL_WARS = ['1973', '1982','2006', '2008', '2012', '2014', '2021']


class Emotions:
    norm = 0
    happy = 2
    sad = 1


class HebrewSongs:
    def __init__(self, path: str = 'data/tagged_data.tsv', normalize_rymes: bool = False):
        self.data: DataFrame = pd.read_csv(path, '\t')
        self.data['decade'] = [int(int(year) / 10) * 10 if type(year) != float and year.isdigit() else 0 for year in self.data["year"]]
        self.data['lyrics_len'] = [len(lyric.split(' ')) for lyric in self.data['lyrics']]
        self.stop_words: List[str] = HebrewSongs.get_stop_words('stopwords.txt')
        self.stop_words_n_grams_2: List[str] = HebrewSongs.get_stop_words('stopwords_ngrams_2.txt')
        self.add_name_in_song()
        self.add_rhymes(normalize=normalize_rymes)
        self.data_hits = self.data[self.data['hit'] == 1]

    def add_colmn(self):
        with open('emotions.txt', 'r') as f:
            emotions_str = f.read().split(',')
            emotions_int = list(map(lambda x: int(x), emotions_str))

        self.data['song_sentiment'] = emotions_int
        self.data.to_csv('data/tagged_data.tsv', sep='\t')

    def get_decade(self, decade: int) -> DataFrame:
        if decade not in DECADES:
            secho(f"[ERROR] decade ({decade}) should be one of the following:", fg="red", bold=True)
            secho(f"        {DECADES}", fg="red")
            exit(1)
        return self.data.loc[(int(self.data['year']) >= decade) & (int(self.data['year']) <= (decade + 9))]

    def get_year(self, year: int) -> DataFrame:
        return self.data.loc[self.data['year'] == year]

    def get_artists_gender(self, hits=False):
        male = 0
        female = 1
        band = 2
        mans = []
        females = []
        bands = []

        for decade in DECADES[:-1]:
            data = self.data[self.data['decade'] == decade]
            if hits:
                data = data[data['hit'] == 1]
            gender_in_hist = data['gender'].value_counts(normalize=True)
            mans.append(gender_in_hist[male] * 100)
            females.append(gender_in_hist[female] * 100)
            bands.append(gender_in_hist[band] * 100)
        p1 = plt.bar(DECADES[:-1], mans, 2, color='dodgerblue', label='man singers')
        p2 = plt.bar(DECADES[:-1], females, 2, bottom=mans, color='deeppink', label='women singers')
        p3 = plt.bar(DECADES[:-1], bands, 2,
                     bottom=np.array(mans) + np.array(females), color='orange', label="bands")
        if not hits:
            plt.title('gender of artists')
        else:
            plt.title('gender of artists in hits')
        plt.xlabel('decade')
        plt.ylabel('precentage of singing gender')
        plt.legend()
        plt.show()

    def get_lyrics(self, df: DataFrame = DataFrame()):
        all_lyrics: Series = self.data["lyrics"] if df.empty else df["lyrics"]
        all_lyrics_str: StringMethods = all_lyrics.str
        words: Series = all_lyrics_str.split()
        words: List[List[str]] = words.values.tolist()
        return [word for i in words for word in i]

    def get_most_common(self, n: int, decade: int = None, use_stopwords: bool = True):
        corpus: List[str] = self.get_lyrics(self.data[self.data['decade'] == decade])
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
        sns.barplot(x=y, y=HebrewSongs.invert_words(x), palette='Paired')
        plt.show()

    def get_emotions_to_wars_plot(self):
        sad_songs = []
        happy_songs = []
        ISRAEL_WARS_plus_1 = [str(int(year)+1) for year in ISRAEL_WARS]
        for year in ['sad_years', 'non_sad']:
            if year == 'sad_years':
                decade_data = self.data[self.data['year'].isin(ISRAEL_WARS+ISRAEL_WARS_plus_1)]
            else:
                decade_data = self.data[-self.data['year'].isin(ISRAEL_WARS+ISRAEL_WARS_plus_1)]

            count_values = decade_data["song_sentiment"].value_counts(normalize=True)
            sad_songs.append(count_values[Emotions.sad])
            happy_songs.append(count_values[Emotions.happy] + count_values[Emotions.norm])

        n = 2
        r = np.arange(n)
        width = 0.25
        plt.bar(r, sad_songs, color='#a3ff58',
                width=width, edgecolor='black',
                label='sad songs')


        plt.ylabel("songs emotion ")
        plt.title("songs sentiment in war years")
        plt.xticks(r + width / 2, ['war years', 'not war years'])
        plt.legend()
        plt.show()

    def get_emotions_plot(self, hits: bool = False):
        male = 0
        female = 1
        band = 2
        sad_songs = []
        happy_songs = []
        normal_songs = []
        if hits:
            self.data = self.data[self.data['hit'] == 1]
        for gender in [male, female, band]:
            decade_data = self.data[self.data["gender"] == gender]
            count_values = decade_data["song_sentiment"].value_counts(normalize=True)
            sad_songs.append(count_values[Emotions.sad])
            happy_songs.append(count_values[Emotions.happy])
            normal_songs.append(count_values[Emotions.norm])

        n = 3
        r = np.arange(n)
        width = 0.2
        plt.bar(r, sad_songs, color='#a3ff58',
                width=width, edgecolor='black',
                label='sad songs')
        plt.bar(r + 2*width, happy_songs, color='#ff58a3',
                width=width, edgecolor='black',
                label='happy songs')
        plt.bar(r + width, normal_songs, color='#D3D3D3',
                width=width, edgecolor='black',
                label='normal songs')
        plt.xlabel("Gender")
        plt.ylabel("songs emotion ")
        plt.title(f"songs sentiment vs gender {'(hits)' if hits else ''}")
        plt.xticks(r + width / 2, ['male', 'female', 'band'])
        plt.legend()
        plt.show()

    def get_song_length_from_years(self):
        self.data['lyrics_len'] = [len(lyric.split(' ')) for lyric in self.data['lyrics']]
        self.data['decade'] = [int(int(year) / 10) * 10 if type(year) != float and year.isdigit() else 0 for year in
                               self.data["year"]]
        length_from_years = self.data[["lyrics_len", "decade"]].groupby("decade").mean()
        x, y = [], []
        for year, length in length_from_years.iterrows():
            if year <= 1960:
                continue
            x.append(str(year))
            y.append(length[0])

        width = 0.4
        barlist = plt.bar(np.arange(len(y)), y, width=width)
        for i, color in enumerate(["#2AC3FF", "#FF2A7C", "#FF632A", "#FFC32A", "#DFFF2A", "#b9e769"]):
            barlist[i].set_color(color)
        plt.xlabel("Decade")
        plt.ylabel("Average length of songs")
        plt.title("decade vs length of songs")
        plt.xticks(np.arange(len(y)), x)
        plt.legend()
        plt.show()

    def get_ngram_most_common(self, n: int, df: DataFrame = DataFrame(), decade=None, ngram_range: tuple = (3, 4), show=True):
        data = self.data
        if decade is not None:
            data = self.data[self.data['decade'] == decade]

        all_lyrics: Series = data["lyrics"] if df.empty else df["lyrics"]

        vec = CountVectorizer(ngram_range=ngram_range, lowercase=True, binary=True).fit(all_lyrics)
        bag_of_words = vec.transform(all_lyrics)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items() 
                      if word not in self.stop_words_n_grams_2]
        top_n_bigrams: List[Tuple[str]] = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
        x, y = map(list, zip(*top_n_bigrams))

        sns.barplot(x=y, y=HebrewSongs.invert_words(x))
        if show:
            plt.title(decade if decade else "")
            plt.xlabel("תורידת")
            plt.show()
        return x

    def uniqe_ngram_per_decade(self, ngram_range=(3,4)):
        decades_words = {}
        for decade in DECADES:
            m = self.get_ngram_most_common(15, decade=decade, ngram_range=ngram_range, show=False)
            decades_words[decade] = m
        uniqe_per_decade = {}
        for decade in DECADES:
            words = decades_words[decade]
            uniwq_words = []
            for word in words:
                seen = False
                for d in DECADES:
                    if d != decade:
                        if word in decades_words[d]:
                            seen = True
                if not seen:
                    uniwq_words.append(word)
            uniqe_per_decade[decade] = uniwq_words
        for k,v in uniqe_per_decade.items():
            secho(f"{k}:  ", fg='blue', bold=True, nl=False)
            secho(f"{', '.join(HebrewSongs.invert_words(v))}", fg='blue')
        return uniqe_per_decade

    def print_number_bits(self):
        for decade in DECADES:
            print(f"decade = {decade}")
            self.get_number_bits(decade=decade)

    def get_number_bits(self, decade):
        lyrics = self.data[self.data['decade'] == decade]['lyrics']
        lyrics_len = 0
        for l in lyrics:
            lyrics_len += len(l.split('\n\n'))
        print(f"lyrics avrage = {lyrics_len / len(lyrics)}")

    def analyze_song_sintiment(self):
        from alephBERT import predict_single_text_with_norm
        songs_sentiment = []
        for _, row in tqdm(self.data.iterrows()):
            lyrics = row['lyrics']
            song_lines = get_songs_lines(lyrics)
            all_song = []
            for line in song_lines:
                prediction = predict_single_text_with_norm(self.bert_classifier, line, 0.3, 0.8)
                all_song.append(prediction)
            if all_song.count(Emotions.sad) / len(all_song) > 0.2:
                song_sentiment = Emotions.sad
            elif all_song.count(Emotions.happy) / len(all_song) > 0.9:
                song_sentiment = Emotions.happy
            else:
                song_sentiment = Emotions.norm
            songs_sentiment.append(song_sentiment)

        self.data['song_sentiment'] = songs_sentiment
        self.data.to_csv('data/tagged_data.tsv', sep='\t')

    def words_more_then(self, num_times=100):
        words_dict = defaultdict(int)
        freq_words = set()
        for _, row in self.data.iterrows():
            lyrics = row['lyrics']

            song_words = lyrics.split(' ')
            for word in song_words:
                words_dict[word] += 1
        for word, num in words_dict.items():
            if num > num_times:
                freq_words.add(word)
        return freq_words

    def add_name_in_song(self, num_times: int = 200):
        name_in_song = []
        freq_words = self.words_more_then(num_times)
        for _, row in self.data.iterrows():
            first_name = row['artist_name'].split()[0]
            lyrics = row['lyrics']
            if first_name not in freq_words:
                name_in_song.append(first_name in lyrics.split(' '))
            else:
                name_in_song.append(False)
        self.data['artist_name_in_song'] = name_in_song

    def plot_name_in_song(self):
        a = self.data[["artist_name_in_song", "decade"]].groupby("decade").mean()
        plt.plot(DECADES, [a['artist_name_in_song'][decade] for decade in DECADES])
        plt.title('artist name in a song')
        plt.xlabel('decade')
        plt.ylabel('num of songs with singer names in the song / total')
        plt.legend()
        plt.show()

    def guess_from_words(self, artists_list, feture):
        learn_feature = feture
        learn_artists = self.data[self.data[learn_feature].isin(artists_list)]
        train, test = train_test_split(learn_artists, test_size=0.2)
        X = train["lyrics"]
        Xtest = test["lyrics"]
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X)
        X_test = vectorizer.transform(Xtest)
        y_test = test[learn_feature]
        y_train = train[learn_feature]
        decision_tree = DecisionTreeClassifier(random_state=0, max_depth=4)
        decision_tree = decision_tree.fit(X_train, y_train)
        tree_text = export_text(decision_tree, feature_names=[HebrewSongs.invert_words(word) for word in vectorizer.get_feature_names()])
        res_pred = decision_tree.predict(X_test)
        score = accuracy_score(y_test, res_pred)
        decision_tree.score(X_test, y_test)

        print(tree_text)
        print(score)

    def split_for_test(self, artists_list, feature_check_name='artist_name'):
        learn_feature = feature_check_name
        know_featurs = ['hit', 'year', 'lyrics_len', 'song_sentiment', "artist_name_in_song", 'artist_name', 'gender']
        # know_featurs = ['hit', 'lyrics_len', 'song_sentiment', 'artist_name']

        if learn_feature in know_featurs:
            know_featurs.remove(learn_feature)
        # gapminder.year.isin(years)
        learn_artists = self.data[self.data[feature_check_name].isin(artists_list)]

        # print(learn_artists['artist_name'].value_counts())
        X = learn_artists[know_featurs]
        train, test = train_test_split(learn_artists, test_size=0.2)
        # X = self.data["lyrics"]
        # vectorizer = CountVectorizer()
        # X_train = vectorizer.fit_transform(X)
        X_train = train[know_featurs]
        X_test = test[know_featurs]
        y_test = test[learn_feature]
        y_train = train[learn_feature]
        return X_train, X_test, y_test, y_train, know_featurs

    def guess_the_artist(self, artists_list, feature_check_name='artist_name'):
        X_train, X_test, y_test, y_train, know_featurs = self.split_for_test(artists_list, feature_check_name)
        decision_tree = DecisionTreeClassifier(random_state=0, max_depth=4)
        decision_tree = decision_tree.fit(X_train, y_train)
        tree_text = export_text(decision_tree, feature_names=know_featurs)
        res_pred = decision_tree.predict(X_test)
        score = accuracy_score(y_test, res_pred)
        decision_tree.score(X_test, y_test)
        print(tree_text)
        # print(score)

    def predict_with_mnb(self, artists_list, feature_check_name='artist_name'):
        X_train, X_test, y_test, y_train, know_featurs = self.split_for_test(artists_list, feature_check_name)
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        mnb_prediction = mnb.predict(X_test)
        y_test = np.array(y_test)
        evaluate(mnb_prediction, y_test)

    def learn_from_lyrics(self):
        self.data['decade'] = [int(int(year) / 10) * 10 if type(year) != float and year.isdigit() else 0 for year in self.data["year"]]
        self.data = self.data[self.data["decade"] >= 1970]
        
        X = self.data["lyrics"]
        y = self.data["decade"]
        
        vectorizer = CountVectorizer()
        mnb = MultinomialNB()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = vectorizer.fit_transform(X_train)
        mnb.fit(X_train, y_train)
        
        X_test = vectorizer.transform(X_test)
        y_test = np.array(y_test)
        mnb_prediction = mnb.predict(X_test)
        
        evaluate(mnb_prediction, y_test)
        return X_train, X_test, y_train, y_test, vectorizer

    def learn_decade(self,learn_feature='gender'):
        X = self.data[['hit', 'decade', 'lyrics_len']]
        # X = self.data["lyrics"]
        vectorizer = CountVectorizer()
        # X_train = vectorizer.fit_transform(X)
        X_train = X
        featurs = ['hit', 'decade', 'lyrics_len']

        y = self.data[learn_feature]
        decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
        decision_tree = decision_tree.fit(X_train, y)
        r = export_text(decision_tree, feature_names=featurs)
        res_pred = decision_tree.predict(X)
        score = accuracy_score(y, res_pred)
        decision_tree.score(X_train, y)
        print(r)
        print(score)
        tree.plot_tree(decision_tree, feature_names=featurs)
        plt.show()

    def write_measure_of_sadness_and_joy(self, bert_classifier):
        from alephBERT import predict_single_text_with_norm
        happiest_scores = []
        saddest_scores = []
        for _, row in tqdm(self.data.iterrows()):
            song_lines = get_songs_lines(row['lyrics'])
            all_song = [predict_single_text_with_norm(bert_classifier, line, 0.3,0.8) for line in song_lines]
            happiest_score = all_song.count(Emotions.happy) / len(all_song)
            saddest_score = all_song.count(Emotions.sad) / len(all_song)

            happiest_scores.append(happiest_score)
            saddest_scores.append(saddest_score)

        self.data['happiest_score'] = happiest_scores
        self.data['saddest_score'] = saddest_scores
        self.data.to_csv('data/tagged_data__new.tsv', sep='\t')

    def add_rhymes(self, normalize: bool = False):
        rhymes = []
        for _, row in tqdm(self.data.iterrows()):
            song_lines = get_songs_lines(row['lyrics'])
            current_rhymes_sequence = 0
            total_rhymes_sequence = 0
            for i in range(1, len(song_lines)):
                last_line = song_lines[i-1].replace(",","").replace(".","").strip()
                current_line = song_lines[i].replace(",","").replace(".","").strip()
                if last_line and current_line and last_line[-2:] == current_line[-2:]:
                    current_rhymes_sequence += 1
                else:
                    total_rhymes_sequence += current_rhymes_sequence
                    current_rhymes_sequence = 0
            rhymes.append(total_rhymes_sequence / (len(song_lines) if normalize else 1))

        self.data['rhymes'] = rhymes

    def plot_rymes(self):
        a = model.data.groupby('decade')["rhymes"].mean()
        x_deta = []
        y_deta = []
        for x,y in a.items():
            if 1970<=float(x)<=2020:
                x_deta.append(y)
                y_deta.append(float(x))
        plt.plot(y_deta,x_deta)
        plt.title(f"Average number of rymes vs decade")
        plt.xlabel(f"Decade")
        plt.ylabel(f"Average number of rymes in song")
        plt.show()


    def get_most_happiest_songs(self,number_of_happy_songs=10):
        data = self.data.sort_values("happiest_score", ascending=False)
        happiest_songs = []
        for _, row in tqdm(data.iterrows()):
            # curr_score = row['happiest_score'] / len(row['lyrics'].split())
            curr_score = row['happiest_score']
            happiest_songs.append((row, curr_score))
        happiest_songs = sorted(happiest_songs, key=lambda x: x[1], reverse=True)
        print(happiest_songs[:number_of_happy_songs])
        return happiest_songs[:number_of_happy_songs]


    def get_most_saddest_songs(self, number_of_sad_songs=10):
        data = self.data.sort_values("saddest_score", ascending=False)
        saddest_songs = []
        for _, row in tqdm(data.iterrows()):
            curr_score = row['saddest_score'] / len(row['lyrics'].split())
            saddest_songs.append((row, curr_score))
        saddest_songs = sorted(saddest_songs, key=lambda x: x[1], reverse=True)
        print(saddest_songs[:number_of_sad_songs])
        return saddest_songs[:number_of_sad_songs]

    @staticmethod
    def get_stop_words(path: str) -> Set[str]:
        with open(path, 'r', encoding='utf8') as f:
            return set(f.read().split('\n'))

    @staticmethod
    def invert_words(words):
        return [w[::-1] for w in words]




def write_test_and_train_csv():
    for data_type in ["test", 'train']:
        dataset = load_dataset("hebrew_sentiment")
        labels = []
        texts = []
        MAX_LEN_ENCODE = 120
        for data in dataset[data_type]:
            text = data["text"].replace(',', '')
            if len(text) < MAX_LEN_ENCODE and int(data["label"]) != 2:
                labels.append(data["label"])
                texts.append(deEmojify(text))

        df = pd.DataFrame({'label': labels, 'text': texts}, columns=['label', 'text'])
        df.to_csv(f'data/{data_type}_sentiment.csv')


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def get_songs_lines(song) -> List[str]:
    song_lines_not_filtered = song.split('\n')
    song_lines = [line for line in song_lines_not_filtered if line != '']
    return song_lines


if __name__ == '__main__':
    # pass
    model = HebrewSongs()
    # model.get_artists_gender(hits=True)
    # model.get_artists_gender()
    # model.get_ngram_most_common(20, ngram_range=(2, 2), decade=1970)
    # for dacade in [1970, 1980, 1990, 2000, 2010, 2020]:
    #     model.get_ngram_most_common(20, ngram_range=(2, 2), decade=dacade, show=False)
    # model.uniqe_ngram_per_decade(ngram_range=(3,4))
    # model.plot_name_in_song()
    # model.get_song_length_from_years()
    # model.plot_rymes()
    # data = model.data.sort_values("rhymes", ascending=False)
    # i = 0
    # for row in data.iterrows():
    #     print(row)
    #     i += 1
    #     if i == 3: 
    #         break
    # model.get_emotions_plot(hits=True)
    # model.get_emotions_to_wars_plot()
    
    # model.get_most_saddest_songs(number_of_sad_songs=5)
    # model.get_most_happiest_songs(number_of_happy_songs=5)
    # for year in [1970, 1980, 1990, 2000, 2010, 2020]:
    #     mean_happiest = model.data.loc[model.data['decade'] == year]["happiest_score"].mean()
    #     mean_saddest = model.data.loc[model.data['decade'] == year]["saddest_score"].mean()
    #     mean_rhymes = model.data.loc[model.data['decade'] == year]["rhymes"].mean()
    #     secho(f"{year}", fg="blue", bold=True)
    #     secho(f"     Happy mean: {mean_happiest}", fg="blue")
    #     secho(f"     Sad mean: {mean_saddest}", fg="blue")
    #     secho(f"     Rhymes mean: {mean_rhymes}", fg="blue")
    
    model.learn_from_lyrics()
    # model.guess_the_artist(['הדג נחש' ,'שרית חדד' ,'כוורת' ,'עומר אדם' ,'אייל גולן' ,'נועה קירל'])

    


