
# # SONG CLUSTERING


# ## Data Preprocessing


# We begin by importing the necessary libraries, classes, and functions required for the following code.


from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(1)


#FIXME
# Eventually, this function will return the data set with dummy encoded values.
# For now, this process is shown in the following chunks.

def data_transform(data_frame):
    data_frame = data_frame.rename(columns={"mode": "modes"})
    genre_encoder = pd.get_dummies(data_frame.genre)
    key_encoder = pd.get_dummies(data_frame.key)
    mode_encoder = pd.get_dummies(data_frame.modes)
    data_frame.drop(['genre','artist_name','track_name','popularity','key','time_signature', 'modes'], axis=1)
    data_frame.join([genre_encoder, key_encoder, mode_encoder])
    return data_frame


# Using a kaggle dataset of roughly 274000 songs and their associated features, we create a pandas dataframe which we will later use to create song clusters based on feature similarity.


spotify_data = pd.read_csv('/Users/jacobgerhard/Documents/Projects/SpotifySongSorting/SpotifyFeatures.csv')
spotify_data = spotify_data.rename(columns={"mode": "modes"})


# We use dummy encoding for features involving string values. In this case, we generate these dummy encoding variables for the genre, key, and mode of each song, allowing us to use these in our k-means clustering.


spotify_feats = spotify_data
genre_encode = pd.get_dummies(spotify_feats.genre)
key_encode = pd.get_dummies(spotify_feats.key)
mode_encode = pd.get_dummies(spotify_feats.modes)


spotify_feats = spotify_feats.drop('genre', axis=1)
spotify_feats.head()



scaled_feats = MinMaxScaler().fit_transform([
    spotify_feats['acousticness'].values,
    spotify_feats['danceability'].values,
    spotify_feats['duration_ms'].values,
    spotify_feats['energy'].values,
    spotify_feats['instrumentalness'].values,
    spotify_feats['liveness'].values,
    spotify_feats['loudness'].values,
    spotify_feats['speechiness'].values,
    spotify_feats['tempo'].values,
    spotify_feats['valence'].values,
])

spotify_feats[['acousticness','danceability','duration_ms',
               'energy','instrumentalness','liveness','loudness',
               'speechiness','tempo','valence']] = scaled_feats.T



spotify_feats = spotify_feats.drop(['artist_name','track_name',
                                    'popularity','key','time_signature',
                                    'modes'], axis=1)
spotify_feats = spotify_feats.join([key_encode, mode_encode])


# Our modified data set after manipulation is shown below.


spotify_feats.head()


# ## Data Clustering


# We begin the clustering step by creating a pipeline using k-means clustering. We set the number of clusters to 25 since that seems to give us a relatively flat histogram (we will increase this later, but this value represents the number of "song types" that exist). We also define the variable `spotify_clus` so that the cluster pipeline will only run using the numberic values of `spotify_feats`.


spotify_clus = spotify_feats.select_dtypes(np.number)


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20, verbose=2,
                                    n_jobs=4))], verbose=True)


song_cluster_pipeline.fit(spotify_clus)
song_cluster_labels = song_cluster_pipeline.predict(spotify_clus)
spotify_data['cluster_label'] = song_cluster_labels
spotify_feats['cluster_label'] = song_cluster_labels



plt.hist(spotify_data['cluster_label'], len(np.unique(spotify_data['cluster_label'])))
plt.show()


# Now, we want to visualize the clusters on a two dimensional plane. We do this using principal component analysis and analyzing the two features with the largest variance within the data set. We create a new pipeline involving `PCA()`, and assign a new variable to the transformed data set `song_embedding`. 

'''
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(spotify_clus)

projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = spotify_data['track_name']
projection['cluster'] = spotify_data['cluster_label']


# The plot of the PCA with two components is below. We notice that the songs are mainly grouped into two main differences (possibly Major and Minor), with one specific cluster appearing further to the right of the others.



fig = px.scatter(projection, x='x', y='y', hover_data=['x', 'y', 'title'], color = 'cluster')
fig.show()
'''

# Looking at a certain cluster of the dataset, we notice that the genre's of all the songs are the same, giving us an idea that we are correctly clustering similar songs.



# Now we can move onto storing our data sets that include cluster labels, which will be helpful in creating a supervised machine learning problem, where cluster numbers are the response variable.


model_data_named = spotify_data
model_data_num = spotify_feats

model_data_num.to_csv('/Users/jacobgerhard/Documents/Projects/SpotifySongSorting/Clusters.csv')

print('Clusters have been assigned to the training data')