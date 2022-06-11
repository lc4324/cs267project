import pandas as pd
import numpy as np
import random
import re

from tbn.tbn import TBN
from tbn.node import Node
from tac import TAC
from train.data import evd_id2col, mar_id2mar

# experiment settings
DATA_SIZE = 100
NUM_LABELS = 4
NUM_ALBUM_STATES = 4
NUM_GENRE_STATES = 4
NUM_ARTIST_STATES = 4
TRAIN_TEST_SPLIT = 0.8

CSV_PATH = 'data.csv'


def read_data(filepath, data_size):
    # read dataset
    dt = {'album_favorites': 'Int64', 'album_title': 'category', 
    'artist_favorites': 'Int64', 'artist_name': 'category', 'track_favorites':'Int64', 'track_genre_top': 'category',
    'track_genres': 'object', 'track_genres_all':'object', 'track_interest':'Int64', 'track_listens': 'Int64',
    'track_title': 'object' }
    #csv_path = '/content/drive/My Drive/Colab Notebooks/data.csv'
    data_frame = pd.read_csv(filepath, dtype= dt)
    data_frame = data_frame[:data_size]
    data_frame = convert_to_level(data_frame)
    print(data_frame.dtypes)
    print("data size: ", data_size)
    return data_frame

def get_level(q75, q50, q25, val): 
    if val > q75: return 3
    if val > q50: return 2
    if val > q25: return 1
    return 0

#split the numerical value into 4 levels.
def convert_to_level(data_frame):
  track_fav_q75 = data_frame['track_favorites'].quantile(0.75)
  track_fav_q50 = data_frame['track_favorites'].quantile(0.50)
  track_fav_q25 = data_frame['track_favorites'].quantile(0.25)
  data_frame['track_favorites_level'] = data_frame.apply(lambda row: get_level(track_fav_q75,track_fav_q50,track_fav_q25, row['track_favorites']), axis = 1)
  return data_frame

# extract relations
def clean(str):
        pattern = re.compile('[^a-zA-Z]+')
        str = pattern.sub('', str).lower()
        return str

def build_dataset(data_frame):
    songs = set()
    albums = set()
    artists = set()
    genres = set()
    data_rows = {}
    
    for index, row in data_frame.iterrows():
        # for each record
        song = str(row['track_title'])
        song = clean(song)
        if song in songs:
            print(f"repeated song: {song}")
            continue
            #raise ValueError(f"Repeated song: {song}." )
        songs.add(song)
        # get album
        album = str(row['album_title'])
        album = clean(album) + " (A)"
        albums.add(album)
        # get artist
        artist = str(row['artist_name'])
        artist = clean(artist) + " (P)"
        artists.add(artist)
        # num_relations += 1
        # get genre
        genre = str(row['track_genre_top']) # TODO: NaN genre?
        genre = clean(genre) + " (G)"
        genres.add(genre)
        # get label
        label = int(row['track_favorites_level'])
        if not (label >= 0 and label <= 3):
            raise ValueError(f"invalid label {label}")
        row2 = {"album":album, "artist":artist, "genre":genre, "label":label}
        data_rows[song] = row2

    num_songs = len(songs)
    num_albums = len(albums)
    num_genres = len(genres)
    num_artists = len(artists)
    songs = list(songs)
    random.shuffle(songs)
    train_data_size = int(len(songs) * TRAIN_TEST_SPLIT)
    test_data_size = len(songs) - train_data_size
    train_songs, test_songs = songs[:train_data_size], songs[train_data_size:]
    train_dataset = [data_rows[song] for song in train_songs]
    test_dataset = [data_rows[song] for song in test_songs]
    
    print(f"train size: {train_data_size}, test_size: {test_data_size}")
    return train_dataset, test_dataset

def transform_dataset_for_toy_BN(dataset, bn):
    num_albums = bn.node("album").card
    num_genres = bn.node("genre").card

    def album_name_to_id(x):  return bn.node("album").values.index(x)
    def genre_name_to_id(x):  return bn.node("genre").values.index(x)

    for i,row in enumerate(dataset):
        album, genre, label = row["album"], row["genre"], row["label"]
        dataset[i] = [album_name_to_id(album), genre_name_to_id(genre), label]
        
    dataset = np.array(dataset)
    evidence = dataset[:,:-1]
    query = dataset[:,-1]
    print(f"evidence size: {evidence.shape}")
    print("evidence: ", evidence)
    evidence = evd_id2col(evidence, cards=[num_albums, num_genres])
    query = mar_id2mar(query, NUM_LABELS)
    return evidence, query


def extract_relations(data_frame):
    songs = set()
    album_to_songs = {}
    artist_to_songs = {} 
    genre_to_songs = {}
    song_to_label = {}
    num_relations = 0

    for index, row in data_frame.iterrows():
        # for each record
        song = str(row['track_title'])
        song = clean(song)
        if song in songs:
            print(f"repeated song: {song}")
            continue
            #raise ValueError(f"Repeated song: {song}." )
        songs.add(song)
        # get album
        album = str(row['album_title'])
        album = clean(album) + "(A)"
        if not album in album_to_songs:
            album_to_songs[album] = []
        album_to_songs[album].append(song)
        num_relations += 1
        # get artist
        artist = str(row['artist_name'])
       # artist = clean(artist) + "(P)"
        if not artist in artist_to_songs:
            artist_to_songs[artist] = []
        artist_to_songs[artist].append(song)
        # num_relations += 1
        # get genre
        genre = str(row['track_genre_top']) # TODO: NaN genre?
        genre = clean(genre) + "(G)"
        if not genre in genre_to_songs:
            genre_to_songs[genre] = []
        genre_to_songs[genre].append(song)
        num_relations += 1
        # get label
        label = int(row['track_favorites_level'])
        if not (label >= 0 and label <= 3):
            raise ValueError(f"invalid label {label}")
        song_to_label[song] = label

    num_songs = len(songs)
    num_albums = len(album_to_songs)
    num_genres = len(genre_to_songs)
    num_artists = len(artist_to_songs)
    print(f"song: {num_songs}, album: {num_albums}, genre: {num_genres}, artist: {num_artists}, relations:{num_relations}")
    return songs, album_to_songs, artist_to_songs, genre_to_songs, song_to_label

    



# build BN model based on relational data
def build_BN(song_titles, album_to_songs, genre_to_songs):
    bn = TBN("recommendation")
    # add nodes
    # each album is a root node
    album_to_node = {}
    for album in album_to_songs.keys():
        name = album
        values = ["%d"%i for i in range(NUM_ALBUM_STATES)]
        node = Node(name, values=values, parents=[])
        album_to_node[album] = node
        bn.add(node)

    # each genre is a root node 
    genre_to_node = {}
    for genre in genre_to_songs.keys():
        name = genre
        values = ["%d"%i for i in range(NUM_GENRE_STATES)]
        node = Node(name, values=values, parents=[])
        genre_to_node[genre] = node
        bn.add(node)

    # each album is the parent of (a proxy of) its songs
    # songs belonged to the same album has the same CPT
    song_to_node_a = {}
    for album, songs in album_to_songs.items():
        for song in songs:
            name = song + "_a"
            values = ["%d"%i for i in range(NUM_LABELS)]
            parent = album_to_node[album]
            cpt_tie = f"belong_to_album_{album}"
            node = Node(name, values=values, parents=[parent], cpt_tie=cpt_tie)
            song_to_node_a[song] = node
            bn.add(node)

    # each genre is the parent of (a proxy of) its songs
    # songs belonged to the same genre has the same CPT
    song_to_node_g = {}
    for genre, songs in genre_to_songs.items():
        for song in songs:
            name = song + "_g" 
            values = ["%d"%i for i in range(NUM_LABELS)]
            parent = genre_to_node[genre]
            cpt_tie = f"belong_to_genre_{genre}"
            node = Node(name, values=values, parents=[parent], cpt_tie=cpt_tie)
            song_to_node_g[song] = node
            bn.add(node)

    # for each song, add a leaf node that is a child of its album proxy node and genre proxy node
    for song in song_titles:
        name = song
        values = ["%d"%i for i in range(NUM_LABELS)]
        song_a = song_to_node_a[song]
        song_g = song_to_node_g[song]
        cpt_tie = f"ensemble"
        node = Node(name, values=values, parents=[song_a, song_g], cpt_tie=cpt_tie)
        song_to_node_g[song] = node
        bn.add(node)

    return bn

# build a toy BN model 
def build_toy_BN(song_titles, album_to_songs, genre_to_songs):
    albums = list(album_to_songs.keys())
    genres = list(genre_to_songs.keys())
    bn = TBN("toy-recommendation")
    albumNode = Node(name="album", values=albums, parents=[])
    bn.add(albumNode)
    genreNode = Node(name="genre", values=genres, parents=[])
    bn.add(genreNode)
    labels = [str(i) for i in range(NUM_LABELS)]
    hidden_node_a = Node(name="hidden_a", values=labels, parents=[albumNode])
    bn.add(hidden_node_a)
    hidden_node_g = Node(name="hidden_g", values=labels, parents=[genreNode])
    bn.add(hidden_node_g)
    output_node = Node(name="output", values=labels, parents = [hidden_node_a, hidden_node_g])
    bn.add(output_node)
    return bn

from sklearn.metrics import accuracy_score, f1_score

def get_labels(dist):
    return np.argmax(dist, axis=1)

def train_toy_BN_from_dataset(csv_path, data_size):
    data_frame = read_data(csv_path, data_size)
    train_data, test_data = build_dataset(data_frame)
    # build BN model
    songs, album_to_songs, artist_to_songs, genre_to_songs, song_to_label = extract_relations(data_frame)
    bn = build_toy_BN(songs, album_to_songs, genre_to_songs)
    bn.dot(fname=f"tbn_toy.gv", view=False)
    # transform dataset for BN inputs
    train_evd, train_mar = transform_dataset_for_toy_BN(train_data, bn)
    test_evd, test_mar = transform_dataset_for_toy_BN(test_data, bn)
    # train circuits
    ac = TAC(bn, inputs=["album", "genre"], output="output", trainable=True)
    ac.fit(train_evd, train_mar, loss_type="CE", metric_type="CE")
    train_acc = ac.metric(train_evd, train_mar,metric_type="CA")
    test_acc = ac.metric(test_evd, test_mar,metric_type="CA")
    print("train acc: {:5f} test acc: {:5f}".format(train_acc, test_acc))

    test_mar_pred = ac.evaluate(test_evd)
    test_y = get_labels(test_mar)
    test_y_pred = get_labels(test_mar_pred)
    acc = accuracy_score(test_y, test_y_pred)
    f1_micro = f1_score(test_y, test_y_pred, average='micro')
    f1_macro = f1_score(test_y, test_y_pred, average='macro')
    print("acc: {:5f} micro f1: {:5f} macro f1: {:5f}".format(acc, f1_micro, f1_macro))
    



    
def gen_BN_from_dataset(csv_path, data_size):
    data_frame = read_data(csv_path, data_size)
    songs, album_to_songs, artist_to_songs, genre_to_songs, song_to_label = extract_relations(data_frame)
    bn = build_BN(songs, album_to_songs, genre_to_songs)
    #bn.dot(fname=f"tbn{DATA_SIZE}.gv", view=False)
    return bn, list(songs)

def compile_bn(bn, songs):
    evidence = songs[:-1]
    Q = songs[-1]
    ac = TAC(bn, inputs=evidence, output=Q, trainable=True)
    return ac
    


if __name__ == '__main__':
    bn, songs = gen_BN_from_dataset(CSV_PATH, DATA_SIZE)
    ac = compile_bn(bn, songs)
    #train_toy_BN_from_dataset(CSV_PATH, DATA_SIZE)

    



    


        
    

