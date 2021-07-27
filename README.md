<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/logo.png?raw=true" width="840px" alt="hebrew-songs"/>
</div>  
&nbsp;  

## :dart: Goal
Our main goal is to ...  
&nbsp;  

## :bar_chart: Results
#### :male_sign: Gender
Here we check the gender of the artists over the decades
```bash
from analyze import HebrewSongs

model = HebrewSongs()
model.get_artists_gender()
```  
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_artists_gender.png?raw=true" alt="get-artists-gender"/>
</div>  
&nbsp;  

#### :clipboard: Most common words
Here we check the most common words (single word)  
```bash
from analyze import HebrewSongs

model = HebrewSongs()
model.model.get_most_common(20)
```  
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_most_common.png?raw=true" alt="get-most-common"/>
</div>  
We can split it to differen decades:  
```bash
model.model.get_most_common(20, decade=1970)
``` 
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_most_common_1970.png?raw=true" alt="get-most-common-1970"/>
</div>  
```bash
model.model.get_most_common(20, decade=2010)
``` 
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_most_common_2010.png?raw=true" alt="get-most-common-2010"/>
</div>  
&nbsp;  

We also can do it for ngrams.  
Here we use the most common of secuences in length between 3 to 4.  
```bash
model.model.get_ngram_most_common(20, ngram_range=(3, 4))
```  
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_ngram_most_common_3_4.png?raw=true" alt="get-ngram-most-common-3-4"/>
</div>  
And for specific decade:  
```bash
model.model.get_ngram_most_common(20, decade=2000, ngram_range=(3, 4))
```  
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_ngram_most_common_3_4_2000.png?raw=true" alt="get-ngram-most-common-3-4-2000"/>
</div> 
&nbsp;  

#### :monkey_face: Artist's name in songs
Here we check the frequencies of artists that say their own name in their songs.
```bash
from analyze import HebrewSongs

model = HebrewSongs()
model.plot_name_in_song()
```  
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/plot_name_in_song.png?raw=true" alt="get-artists-gender"/>
</div>  
&nbsp;  

#### :sparkling_heart: Emotions
Here we check if the songs are sad or happy.  
We fine tunning a pre-traning AlephBERT model, for sentiment analasis.  
So we have model that get a song and return if the song is happy or sad.  
We also create some graph which show the different between females, mans and bands:  
```bash
from analyze import HebrewSongs

model = HebrewSongs()
model.get_emotions_plot()
```  
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_emotions_plot.png?raw=true" alt="get-emotions-plot"/>
</div>  
&nbsp;  

#### :scroll: Length of songs per decade
Here we check the length of the songs per decades.  
Its interesting that new songs are much more longer.  
```bash
from analyze import HebrewSongs

model = HebrewSongs()
model.get_song_length_from_years()
```  
<div align="center">
<img src="https://github.com/shaharjacob/hebrew-songs/blob/main/images/get_song_length_from_years.png?raw=true" alt="get-song-length-from-years"/>
</div>  
&nbsp;  

## :clipboard: References
- **Shironet**: https://shironet.mako.co.il/  
- **Wikipedia**: [https://he.wikipedia.org/wiki/מצעד_הפזמונים_העברי_השנתי](https://he.wikipedia.org/wiki/%D7%9E%D7%A6%D7%A2%D7%93_%D7%94%D7%A4%D7%96%D7%9E%D7%95%D7%A0%D7%99%D7%9D_%D7%94%D7%A2%D7%91%D7%A8%D7%99_%D7%94%D7%A9%D7%A0%D7%AA%D7%99)  
- **AlephBERT**: https://huggingface.co/onlplab/alephbert-base  