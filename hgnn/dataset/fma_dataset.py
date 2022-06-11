import dgl
from dgl.data import DGLDataset
import torch
import os

import pandas as pd
import numpy as np
#from transformers import Trainer
# from collections import Counter
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import networkx as nx
import matplotlib.pyplot as plt
import dgl

import warnings
warnings.filterwarnings('ignore')

import dgl


def get_level(q75, q50, q25, val):
    if val >= q75: return 4
    if val >= q50: return 3
    if val > q25: return 2
    return 1


def read_data_and_generate_graph_new():

    dt = {'album_favorites': 'Int64', 'album_title': 'category',
          'artist_favorites': 'Int64', 'artist_name': 'category', 'track_favorites': 'Int64',
          'track_genre_top': 'category',
          'track_genres': 'object', 'track_genres_all': 'object', 'track_interest': 'Int64',
          'track_listens': 'Int64',
          'track_title': 'object'}
    csv_path = './openhgnn/dataset/data.csv'
    # csv_path = './data.csv'
    data_frame = pd.read_csv(csv_path, dtype=dt)

    data_frame = data_frame.drop(data_frame[data_frame['album_title'].isna()].index)
    data_frame = data_frame.drop(data_frame[data_frame['track_genres_all']=='[]'].index)
    # data_frame = data_frame = data_frame.drop(data_frame[data_frame['artist_favorites'] == -1].index)
    # data_frame = data_frame = data_frame.drop(data_frame[data_frame['album_favorites'] == -1].index)
    # print(f"dataset size: {len(data_frame)}")
    track_fav_q75 = data_frame['track_favorites'].quantile(0.75)
    track_fav_q50 = data_frame['track_favorites'].quantile(0.50)
    track_fav_q25 = data_frame['track_favorites'].quantile(0.25)
    data_frame['track_favorites_level'] = data_frame.apply(
        lambda row: get_level(track_fav_q75, track_fav_q50, track_fav_q25, row['track_favorites']), axis=1)

    track_int_q75 = data_frame['track_interest'].quantile(0.75)
    track_int_q50 = data_frame['track_interest'].quantile(0.50)
    track_int_q25 = data_frame['track_interest'].quantile(0.25)
    data_frame['track_interest_level'] = data_frame.apply(
        lambda row: get_level(track_int_q75, track_int_q50, track_int_q25, row['track_interest']), axis=1)

    album_fav_q75 = data_frame['album_favorites'].quantile(0.75)
    album_fav_q50 = data_frame['album_favorites'].quantile(0.50)
    album_fav_q25 = data_frame['album_favorites'].quantile(0.25)
    data_frame['album_favorites_level'] = data_frame.apply(
        lambda row: get_level(album_fav_q75, album_fav_q50, album_fav_q25, row['album_favorites']), axis=1)

    track_lis_q75 = data_frame['track_listens'].quantile(0.75)
    track_lis_q50 = data_frame['track_listens'].quantile(0.50)
    track_lis_q25 = data_frame['track_listens'].quantile(0.25)
    data_frame['track_listens_level'] = data_frame.apply(
        lambda row: get_level(track_lis_q75, track_lis_q50, track_lis_q25, row['track_listens']), axis=1)
    data_frame['track_listens_level'] = MinMaxScaler().fit_transform(
        np.array(data_frame['track_listens_level']).reshape(-1, 1))

    graph_data = {}
    album_node = {}
    artist_node = {}
    genre_node = {}
    track_node = {}
    node_id = 0
    album_favorites = []

    albums = []
    artists = []
    genres = []
    tracks = []
    levels = []
    listens = []
    interests = []
    node_features = []
    cnt = 0

    genres_all = []
    track_genres_all = []

    for index, row in data_frame.iterrows():
        album = row['album_title']
        if not album in album_node:
            album_node[album] = len(album_node)
            album_favorites.append(row['album_favorites_level']-1)
        albums.append(album_node[album])

        artist = row['artist_name']
        if not artist in artist_node:
            artist_node[artist] = len(artist_node)
        artists.append(artist_node[artist])

        genre = row['track_genre_top']
        if not genre in genre_node:
            genre_node[genre] = len(genre_node)
        genres.append(genre_node[genre])

        convert_to_list = row['track_genres_all'][1:-1].split(',')
        genres_all_tmp = [int(ele) for ele in convert_to_list]
        genres_all.extend(genres_all_tmp)
        track_genres_all.extend([cnt]*len(genres_all_tmp))

        tracks.append(cnt)
        cnt += 1

        levels.append(row['track_favorites_level']-1)
        #listens.append(row['track_listens_level'])
        listens.append(row['track_listens_level'])
        interests.append(row['track_interest_level'])

    # Create a heterograph
    graph_data = {
        ('artist', 'creates', 'album'): (torch.tensor(artists), torch.tensor(albums)),
        ('album', 'is created by', 'artist'): (torch.tensor(albums), torch.tensor(artists)),
        ('album', 'contains', 'track'): (torch.tensor(albums), torch.tensor(tracks)),
        ('track', 'is in', 'album'): (torch.tensor(tracks), torch.tensor(albums)),
        ('track', 'belongs to', 'genre'): (torch.tensor(tracks), torch.tensor(genres)),
        ('genre', 'is a property of', 'track'): (torch.tensor(genres), torch.tensor(tracks)),
        # ('genre', 'is a property of', 'track'): (torch.tensor(genres_all), torch.tensor(track_genres_all)),
        # ('track', 'belongs to', 'genre'): (torch.tensor(track_genres_all), torch.tensor(genres_all)),
    }
    g = dgl.heterograph(graph_data)

    print(len(album_favorites))
    print(max(albums))

    print(g)

    g.nodes['track'].data['labels'] = torch.tensor(levels)
    return g


class FMADataset(DGLDataset):
    def __init__(self):
        super().__init__(name='fma')

    def process(self):
        self.graph = read_data_and_generate_graph_new()
        self.predict_category = 'track'
        self.num_classes = 4


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

