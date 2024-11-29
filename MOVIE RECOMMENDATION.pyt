import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
names = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
ratings_name = ['userId', 'movieId', 'rating', 'timestamp']
movies = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.item", sep="|", names=names, encoding = "latin-1")
ratings = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.data", sep= "\t", names=ratings_name)
movies
ratings
movies.info()
print(movies.columns)   # Check columns in movies DataFrame
print(ratings.columns)  # Check columns in ratings DataFrame

df = pd.merge(movies, ratings, right_on="movieId", left_on="movie_id")

df
# merge movies and ratings. as the columns are named different we use left_on and right_on 
df = pd.merge(movies, ratings, left_on="movie_id", right_on="movieId")

df
#pivot_table is function (tool) here. It is a kind of tool which will summarize my data from the larger table and then group by;
#and aggregate in to your value into the new particular table.(same as in .xlsx we have).  It makes new data.
user_ratings = df.pivot_table(index="userId", columns="title", values="rating")
#943 users
user_ratings
#cannot pass this NaN value to ML. As NaN - no rating i.e 0. so we are changing NaN to 0.Now 0 is a floating value.
user_ratings = user_ratings.fillna(0)
user_ratings
###cosine Similarity
### kmeans
# WCSS? it's sum of the square  distance  between the points in the cluster znd the clusters Android (cluster- combine)
#distance between centroid and point (euclidean distance) we try to find; with that distance we find sum of square of cluster; 
#with that cluster sum of square we will do summation of 1 st cluster and 2 nd cluster;Now together whatever the cluster numbers we have is WCSS or the inertia;
# So on y-axis it's WCSS and on x-axis it will be the number of clusters you want to experiment;
#And wherever we could see the elbow kind of design we can say that this many number of clusters we need for our model. 
#To make the data more clear we do this and always remember "the movie have to find out the user"- problem Statement
movie_features = df.groupby("movieId").agg({"rating":["mean", "count"],
                                             "userId":"count"}).reset_index()
movie_features
movie_features.columns = ["movieId", "avg_rating", "rating_count", "user_account"]
movie_features
geners = movies['genres'].str.get_dummies("|")
