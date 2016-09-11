import json
import codecs
import unidecode
import string
import vincent
import pandas
import folium
from vincent import AxisProperties, PropertySet, ValueRef
from tweetokenize import Tokenizer
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
import numpy as np

import matplotlib.pyplot as plt

def analyze_terms(data_path):
    # GLOBALS
    tweet_tkzr = Tokenizer(usernames=False)
    tweet_count = Counter()
    hashtag_count = Counter()
    bigram_count = Counter()
    username_count = Counter()
    stopwords_set = stopwords.words('english') + list(string.punctuation)
    total_tweets = 0
    tweet_len_list = []
    with codecs.open(data_path, encoding='utf-8') as data_file:
        for line in data_file.readlines():
            line = line.strip()
            if len(line) > 0 :
                tweet = json.loads(line)
                if 'text' in tweet:
                    unidecoded_tweet = unidecode.unidecode_expect_nonascii(tweet['text'])
                    total_tweets += 1
                    tweet_len_list.append(len(tweet['text']))
                    tokens = [token for token in tweet_tkzr.tokenize(unidecoded_tweet) if token not in stopwords_set]
                    tweet_count.update(tokens)
                    bigram_count.update(bigrams(tokens))
                    hashtag_tokens = [token for token in tokens if token.startswith(('#'))]
                    hashtag_count.update(hashtag_tokens)
                    username_tokens = [token for token in tokens if token.startswith(('@'))]
                    username_count.update(username_tokens)

    row_format = "{:>32}{:>10}"
    # BIGRAM
    for term, freq in tweet_count.most_common(25):
        bigram = ' '.join(term)
        print row_format.format(bigram, freq)

    term_freq = tweet_count.most_common(25)
    terms, freq = zip(*term_freq)
    data = {'data': freq, 'x':terms}
    bar = vincent.Bar(data, iter_idx='x')
    axProp = AxisProperties(
        labels=PropertySet(angle=ValueRef(value=270)))
    bar.axes[0].properties = axProp
    bar.to_json('term_freq.json')

    # HASHTAG
    print ''
    for term, freq in hashtag_count.most_common(25):
        print row_format.format(term, freq)

    hashtag_freq = hashtag_count.most_common(25)
    hashtags, freq = zip(*hashtag_freq)
    data = {'data': freq, 'x': hashtags}
    bar = vincent.Bar(data, iter_idx='x')
    axProp = AxisProperties(
        labels=PropertySet(angle=ValueRef(value=270)))
    bar.axes[0].properties = axProp
    bar.to_json('hashtag_freq.json')

    # USERNAMES
    print ''
    for term, freq in username_count.most_common(25):
        print row_format.format(term, freq)

    username_freq = username_count.most_common(25)
    usernames, freq = zip(*username_freq)
    data = {'data': freq, 'x': usernames}
    bar = vincent.Bar(data, iter_idx='x')
    axProp = AxisProperties(
        labels=PropertySet(angle=ValueRef(value=270)))
    bar.axes[0].properties = axProp
    bar.to_json('username_freq.json')

def analyze_correlation(data_path):
    pass

def analyze_timeseries(data_path, search_term):
    tweet_tkzr = Tokenizer(usernames=False)
    stopwords_set = stopwords.words('english') + list(string.punctuation)
    dates_search_term = []
    with codecs.open(data_path, encoding='utf-8') as data_file:
        for line in data_file.readlines():
            line = line.strip()
            if len(line) > 0:
                tweet = json.loads(line)
                if 'text' in tweet:
                    unidecoded_tweet = unidecode.unidecode_expect_nonascii(tweet['text'])
                    tokens = [token for token in tweet_tkzr.tokenize(unidecoded_tweet) if token not in stopwords_set]
                    if search_term in tokens:
                        dates_search_term.append(tweet['created_at'].strip())

    ones = [1] * len(dates_search_term)
    idx = pandas.DatetimeIndex(dates_search_term)
    search_term_series = pandas.Series(ones, index=idx)
    per_minute = search_term_series.resample('1Min', how='sum').fillna(0)

    plt.plot(search_term_series)
    plt.show()

    time_chart = vincent.Line(search_term_series)
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.to_json('BLM_time_chart.json')

def analyze_geolocation(data_path):
    geoloc_count = 0
    tweet_count = 0
    with codecs.open(data_path, encoding='utf-8') as data_file:
        geo_data = {
            "type": "FeatureCollection",
            "features": []
        }
        for line in data_file.readlines():
            line = line.strip()
            if len(line) > 0:
                tweet_count += 1
                tweet = json.loads(line)
                if 'coordinates' in tweet and tweet['coordinates'] is not None:
                    geoloc_count += 1
                    geo_json_feature = {
                        "type": "Feature",
                        "geometry": tweet['coordinates']['coordinates'],
                        "properties": {
                            "text": tweet['text'],
                            "created_at": tweet['created_at']
                        }
                    }
                    geo_data['features'].append(geo_json_feature)
    # Save geo data
    with open('geo_data.json', 'w') as fout:
        fout.write(json.dumps(geo_data, indent=4))

    print 'Geolocated Tweets:  {} / {}'.format(geoloc_count, tweet_count)

def analyze_folium_map(data_path):
    lat_list = []
    long_list = []
    text_list = []

    with codecs.open(data_path, encoding='utf-8') as data_file:
        for line in data_file.readlines():
            line = line.strip()
            if len(line) > 0:
                tweet = json.loads(line)
                if 'coordinates' in tweet and tweet['coordinates'] is not None:
                    lat_list.append(tweet['coordinates']['coordinates'][1])
                    long_list.append(tweet['coordinates']['coordinates'][0])
                if 'text' in tweet:
                    text_list.append(tweet['text'])
                else:
                    text_list.append(u'')

    map = folium.Map(location=[np.mean(lat_list), np.mean(long_list)], zoom_start=6, tiles='Mapbox bright')

    def color(elev):
        return 'blue'

    fg = folium.FeatureGroup(name="Tweet Locations")
    for lat, lon, name in zip(lat_list, long_list, text_list):
        fg.add_child(folium.Marker(location=[lat, lon], popup=(folium.Popup(name)),
                                   icon=folium.Icon(color='blue', icon_color='blue')))
    map.add_child(fg)
    map.add_child(folium.GeoJson(data=open('world_geojson_from_ogr.json'),
                                 name="Population",
                                 style_function=lambda x: {'fillColor': 'green' if x['properties'][
                                                                                       'POP2005'] <= 10000000 else 'orange' if 10000000 <
                                                                                                                               x[
                                                                                                                                   'properties'][
                                                                                                                                   'POP2005'] < 20000000 else 'red'}))
    map.add_child(folium.LayerControl())
    map.inline_map()
    map.save(outfile='folium_map.html')

def analyze_topic_modelling(data_path):
    pass

def analyze_clustering(data_path):
    pass

if __name__=='__main__':
    data_path = '../data/all_data.dat'
    #analyze_terms( data_path )
    #analyze_timeseries( data_path, "police" )
    #analyze_geolocation(data_path)
    analyze_folium_map(data_path)
