# # USER RECOMMENDATION

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

tfmodel = load_model('/Users/jacobgerhard/Documents/Projects/SpotifySongSorting/tfmodel')

'''
# `POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# Convert the response to JSON
auth_response_data = auth_response.json()

# Save the access token
access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

# Base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'

# Track ID from the URI
track_id = '3YXIVMQLRRq2K7kxC7UYx6'

get_playlist = requests.get(BASE_URL + 'playlists/' + playlist_id + '/tracks', headers=headers)

playlist_json = get_playlist.json()
'''


CLIENT_ID = '8ae18f9c35f446f8ba19765bc5656de7'
CLIENT_SECRET = 'a5bca550a7e34646899c3e47eb9a9fb4'

AUTH_URL = 'https://accounts.spotify.com/api/token'


client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


username = 'bocajonian'

# playlists to sort
Spotify_API_Test = '4AwVeRjnV6ntFpD3nx24aX'
Wooden_Ceiling = '55UvHPI0phk6F15XovlNh7'
Jazz_Instrumental_Classics = '1dHJUpqHMRbGN3kpMJDw0T'



def call_playlist(creator, playlist_id):

    playlist_features_list = ["artist","track_name","track_id",
                              "acousticness", "danceability","energy",
                              "key","loudness","mode","speechiness",
                              "instrumentalness","liveness","valence","tempo",
                              "duration_ms","time_signature"]

    playlist_df = pd.DataFrame(columns = playlist_features_list)

    
    user_playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in user_playlist:
  
        playlist_features = {}
        
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[3:]:
            playlist_features[feature] = audio_features[feature]
        

        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)

        
    return playlist_df




playlist_example = call_playlist(username, Jazz_Instrumental_Classics)


mode_dict = {0:'Minor', 1:'Major'}
key_dict = {0:'C',1:'C#',2:'D',3:'D#',4:'E',5:'F',6:'F#',
            7:'G',8:'G#',9:'A',10:'A#',11:'B'}
key_list = list(key_dict.values())

playlist_example = playlist_example.rename(columns = {'mode':'modes'})
playlist_example = playlist_example.replace({'modes': mode_dict}).replace({'key': key_dict})


playlist_feats = playlist_example
key_encode = pd.get_dummies(playlist_feats.key, columns = key_list)
mode_encode = pd.get_dummies(playlist_feats.modes)


for item in key_list:
    if item not in key_encode.columns:
        key_encode.insert(key_list.index(item)+3, item, 0)




scaled_feats = MinMaxScaler().fit_transform([
    playlist_feats['acousticness'].values,
    playlist_feats['danceability'].values,
    playlist_feats['duration_ms'].values,
    playlist_feats['energy'].values,
    playlist_feats['instrumentalness'].values,
    playlist_feats['liveness'].values,
    playlist_feats['loudness'].values,
    playlist_feats['speechiness'].values,
    playlist_feats['tempo'].values,
    playlist_feats['valence'].values,
])

playlist_feats[['acousticness','danceability','duration_ms',
               'energy','instrumentalness','liveness','loudness',
               'speechiness','tempo','valence']] = scaled_feats.T


'''
def data_process(data_frame):
    #genre_encode = pd.get_dummies(data_frame.genre)
    key_encode = pd.get_dummies(data_frame.key)
    mode_encode = pd.get_dummies(data_frame.modes)

    scaled_feats = MinMaxScaler().fit_transform([
        data_frame['acousticness'].values,
        data_frame['danceability'].values,
        data_frame['duration_ms'].values,
        data_frame['energy'].values,
        data_frame['instrumentalness'].values,
        data_frame['liveness'].values,
        data_frame['loudness'].values,
        data_frame['speechiness'].values,
        data_frame['tempo'].values,
        data_frame['valence'].values,
        ])
    data_frame[['acousticness','danceability','duration_ms',
               'energy','instrumentalness','liveness','loudness',
               'speechiness','tempo','valence']] = data_frame.T
    return data_frame

data_process(playlist_feats)
'''


playlist_feats = playlist_example.drop(['artist','track_name',
                                        'key','time_signature',
                                        'modes'], axis=1)
playlist_feats = playlist_feats.join([key_encode, mode_encode])



playlist_feats = playlist_feats.reindex(['track_id','acousticness','danceability',
                        'duration_ms','energy','instrumentalness',
                        'liveness','loudness','speechiness','tempo',
                        'valence','A','A#','B','C','C#','D','D#','E',
                        'F','F#','G','G#','Major','Minor'], axis=1)



playlist_num = playlist_feats.drop('track_id', axis =1)


playlist_num_np = playlist_num.values


new_preds = pd.DataFrame(tfmodel.predict(playlist_num_np)).idxmax(axis=1)


playlist_num['cluster_label'] = new_preds


plt.hist(playlist_num['cluster_label'])
plt.show()
