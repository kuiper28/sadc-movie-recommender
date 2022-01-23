import streamlit as st
import pandas as pd
import numpy as np
import pickle as cPickle

from streamlit.type_util import Key
from data_base import *
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding,RepeatVector,TimeDistributed,Activation,Dropout


import altair as alt
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from neural_colaborative_filtering.model import NCRF


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

@st.cache
def readData():
	ratings_path = "ml-1m/ratings.dat"
	movie_path = "ml-1m/movies.dat"

	names = ['user_id', 'movie_id', 'rating', 'timestamp']
	ratings = pd.read_csv(ratings_path, sep='::', names=names, engine='python', encoding = "ISO-8859-1")

	movies = pd.read_csv(movie_path, sep='::', header=None, usecols=[1], engine='python', encoding = "ISO-8859-1")
	movies.columns = ['title']

	global X, y, dim
	X = ratings[['user_id', 'movie_id']].values
	y = ratings['rating'].values
	dim = (np.unique(np.array(X[:,0])).size, np.unique(np.array(X[:,1])).size)
	# print(X)
	# print(y)
	# print(dim)

	return (X, y, dim, movies)
@st.cache
def getAllMovies():
	return all_movies()

def create_rating_matrix(X, y, dim):
  r = X[:,0]
  c = X[:,1]
  matrix = csr_matrix((y, (r,c)), shape=(dim[0] + 1, dim[0] + 1))
  M = matrix.todense()
  M = M[1:,1:]
  M = np.asarray(M)
  return M

def make_recommendation_for_newuser(item_sim, item_idx, movies, k=5):
    similar_items_df = pd.DataFrame(item_sim).iloc[item_idx, :] 
    similar_items_df = pd.concat([similar_items_df, movies], axis=1)  
    similar_items_df.columns = ['similarity','title']
    similar_items_df = similar_items_df.sort_values(by='similarity',ascending=False)

    # print('Recommended movies for a new user (without rating history), currently looking at movie:', similar_items_df.iloc[0]['title'])
    # print(similar_items_df[1:k+1])
    return(similar_items_df[1:k+1])     

# item_sim = cosine_similarity(H)                     
# make_recommendation_for_newuser(item_sim, item_idx=1, k=5)
# make_recommendation_for_newuser(item_sim, item_idx=20, k=5)
# make_recommendation_for_newuser(item_sim, item_idx=500, k=5)

def getPredictionForSpecificUser(X,y,dim, movies):
	ratings = view_all_ratings(st.session_state.user)
	clean_db = pd.DataFrame(ratings,columns=["username", "rating", "movieid"])
	clean_db["userid"] = 6040

	X_user = clean_db[['userid', 'movieid']].values
	y_user = clean_db['rating'].values

	count = 0
	buff_X = np.copy(X)
	# print(buff_X)
	buff_y = np.copy(y)
	for x in buff_X:
		if x[0] == 6040:
			break
		count += 1

	new_X = buff_X[:count]
	new_y = buff_y[:count]	
	# for x in X_user:
	# 	new_X += x
	# for y in y_user:
	# 	new_y += y

	new_X = np.append(buff_X[:count], X_user, axis = 0)
	new_y = np.append(buff_y[:count], y_user)

	new_dim = (np.unique(np.array(new_X[:,0])).size, np.unique(np.array(new_X[:,1])).size)
	


	M_user = create_rating_matrix(new_X, new_y, new_dim)

	loaded_model = cPickle.load(open("nmf_model.cpickle", 'rb'))
	W = loaded_model.transform(M_user)
	H = loaded_model.components_.T
	P_user = H.dot(W.T).T
	P_user[P_user > 5] = 5.                   
	P_user[P_user < 1] = 1.

	st.subheader("Movie recommendations")
	recommended_movies = make_recommendation_for_an_existing_user(M_user, P_user, movies, 6039, k = 10)
	# print(recommended_movies)
	clean_db = pd.DataFrame(recommended_movies,columns=["title"])
	st.table(clean_db)

@st.cache
def load_model_and_get_predictions(X, y, dim):
  
  loaded_model = cPickle.load(open("nmf_model.cpickle", 'rb'))
  W = loaded_model.transform(create_rating_matrix(X, y, dim))
  H = loaded_model.components_.T
  P = H.dot(W.T).T
  P[P > 5] = 5.                   
  P[P < 1] = 1.
  return P, H

@st.cache
def make_recommendation_for_an_existing_user(initial_rating_matrix, predicted_rating_matrix, movies, user_idx, k=10):
	user_ratings = pd.DataFrame(initial_rating_matrix).iloc[user_idx, :]              
	user_prediction = pd.DataFrame(predicted_rating_matrix).iloc[user_idx,:]
	preferred_movies = np.flip(np.argsort(np.array(user_ratings))[-k:])
	recommended_movies = np.flip(np.argsort(np.array(user_prediction))[-k:])
	# print(movies.iloc[recommended_movies])
	return movies.iloc[recommended_movies]

@st.cache 
def populate_databse():
	users_name = ['user_id', 'gender', 'age', 'ocupation', 'zip-code']
	users = pd.read_csv("ml-1m/users.dat", sep='::', names=users_name)

	movie_names = ["movie_id", "title", "genre"]
	movies = pd.read_csv("ml-1m/movies.dat", sep='::', names=movie_names)

	ratings_columns = ["user_id", "movie_id", "rating", "time_stamp"]
	ratings = pd.read_csv("ml-1m/ratings.dat", sep='::', names=ratings_columns)

	create_usertable()
	for l in users.values:
		add_userdata("username" + str(l[0]), str(l[0]) + "1234", l[1], l[2])
	
	create_movoietable()
	for m in movies.values:
		add_moviedata(str(m[0]), str(m[1]), str(m[2]))

	create_userratings()
	k = 1
	for r in ratings.values:
		add_ratings("username" + str(r[0]), r[1], r[2])
		k += 1



def get_movies_for_a_user():
	ratings_columns = ["user_id", "movie_id", "rating", "time_stamp"]
	ratings = pd.read_csv("ml-1m/ratings.dat", sep='::', names=ratings_columns)
	# print(ratings.values[:10])

def signup_func():
	st.sidebar.subheader("Create An Account")
	new_username = st.sidebar.text_input("User name")
	new_password = st.sidebar.text_input("Password",type='password')
	confirm_password = st.sidebar.text_input('Confirm Password',type='password')
	new_gender = st.sidebar.text_input("Gender")
	new_age = st.sidebar.text_input("Age")

	if new_password == confirm_password:
		st.success("Valid Password Confirmed")
	else:
		st.warning("Password not the same")

	if st.sidebar.checkbox("Sign Up"):
		add_userdata(new_username,new_password, new_gender, new_age)
		st.success("Successfully Created an Account")


def login_func():
	st.sidebar.subheader("Login")
	username = st.sidebar.text_input("Username", on_change=None)
	password = st.sidebar.text_input("Password",type='password', on_change=None)

	if st.sidebar.checkbox("Login"):
		try_login(username, password)
	else:
		st.session_state.user = None



def try_login(username, password):
	result = login_user(username,password)
	if result:
		st.session_state.user = username
	else:
		st.warning("bad")
		st.warning("Incorrect Username/Password")


def run_app_2(X, y, dim, movies, deep=False):
	username = "username1"
	movies_by_username = view_all_movies(username)
	# print(movies_by_username)
	if (deep == True):
		# st.subheader("Movie recommendations")
		recommended_movies = make_prediction_ncf(1, 10, movies)
		# print("Recommended Movies ", recommended_movies)
		clean_db = pd.DataFrame(recommended_movies,columns=["title"])
		# st.table(clean_db)

def run_app_lstm(username, X, y, dim, movies):
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	model = create_model(3953, 3953, 128, 128, 64)
	model.load_weights('./lstm_cf_weights/checkpoint')
	all_movies_db = view_all_movie_ids(username)

	if st.checkbox("Make movie recommendations", key=4) and len(all_movies_db) > 0:
		
		user_movies = [movie[0] for movie in all_movies_db]
		if len(user_movies) > 128:
			user_movies = user_movies[-128:]
		
		user_movies += [0] * (128 - len(user_movies))
		
		model_input = np.array(user_movies)
		count = np.count_nonzero(model_input)
		source = model_input.reshape((1, model_input.shape[0]))
		prediction = model.predict(source, verbose=0)[0]
		user_predictions = np.argpartition(prediction[count-1], -10)[-10:]
		user_predictions = user_predictions.tolist()

		to_remove = []
		for pred in user_predictions:
			if pred in user_movies:
				to_remove.append(pred)

		for remove in to_remove:
			user_predictions.remove(remove)

		pred_titles = []
		for pred in user_predictions:
			# print(get_movie_name(pred))
			pred_titles.append(get_movie_name(pred)[0])

		df = pd.DataFrame(pred_titles)

		st.table(df)



def create_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    # Create the model
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(120, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(tar_vocab, activation='softmax'))
    return model

	
def run_app_ncf(username, X, y, dim, movies):
	movies_by_username = view_all_movies(username)
	# print(movies_by_username)
	if (len(movies_by_username) != 0):
		st.subheader("Movie recommendations")
		recommended_movies = make_prediction_ncf(int(username[-1]), 10, movies)
		clean_db = pd.DataFrame(recommended_movies,columns=["title"])
		st.table(clean_db)
	
	# st.bar_chart(user_df_group)

		
def run_app(username, X, y, dim, movies):
	movies_by_username = view_all_movies(username)
	# print(movies_by_username)
	if (len(movies_by_username) != 0):
		if st.checkbox("Make movie recommendations", key=3):

			predicted_rating_matrix, H = load_model_and_get_predictions(X, y, dim)
			initial_rating_matrix = create_rating_matrix(X,y,dim)

			st.subheader("Movie recommendations")
			recommended_movies = make_recommendation_for_an_existing_user(initial_rating_matrix, predicted_rating_matrix, movies, user_idx=int(username[-1]), k = 10)
			clean_db = pd.DataFrame(recommended_movies,columns=["title"])
			st.table(clean_db)
	else:
		if st.checkbox("Make movie recommendations for new user", key=4):
			st.header("Select the movie you are currently watching!")
			option = select_movie_currently_watching()
			# print(option)
			if (option != None):
				predicted_rating_matrix, H = load_model_and_get_predictions(X, y, dim)
				item_sim = cosine_similarity(H)
				st.subheader("Movie recommendations")
				recommended_movies = make_recommendation_for_newuser(item_sim, option, movies, 10)
				clean_db = pd.DataFrame(recommended_movies,columns=["title"])
				st.table(clean_db)
			# getPredictionForSpecificUser(X,y,dim, movies)
	
def movieTable(username):
	st.header("Add movie ratings")
	addNewMovie()

	if (len(view_all_movie_ids(username)) > 0):
		st.header("Remove movie ratings")
		deleteExistingMovie()

	st.subheader("My watched movies")
	users_result = view_all_movies(username)
	clean_db = pd.DataFrame(users_result, columns=["name", "rating"])
	user_result_by_rating = view_all_movies_group_by_rating(username)
	user_df_group = pd.DataFrame(user_result_by_rating, columns=["rating", "Number of movies"])
	st.table(clean_db)
	# st.bar_chart(user_df_group)

def format(df, id):
	temp = df[df["id"] == id]
	return temp["title"].values[0]

def addNewMovie():
	col1, col2, col3= st.columns(3)
	
	new_db = pd.DataFrame(getAllMovies(), columns=["title", "id"])

	ratings = [1,2,3,4,5]
	
	new_movie_id = col1.selectbox("Movie", options = new_db["id"], format_func = lambda x: format(new_db, x), key = 4)
	new_rating = col2.selectbox("Rating", ratings, key = 5)

	
	if col3.button("Add review"):
		movies = getAllMovies()
		user_movies = view_all_movies(st.session_state.user)
		ok = 1
		for movie in user_movies:
			print(movie[0])
			print(get_movie_id_from_list(movies, movie[0]), new_movie_id)
			if get_movie_id_from_list(movies, movie[0]) == new_movie_id:
				st.warning("Movie was already added")
				ok = 0
				break
		if ok: 
			st.success("Added movie in list")
			add_ratings(st.session_state.user, new_movie_id, new_rating)
		else:
			st.success("Updated movie rating")
			delete_ratings(st.session_state.user, new_movie_id)
			add_ratings(st.session_state.user, new_movie_id, new_rating)

def get_movie_id_from_list(movies, movie):
	for m in movies:
		if (m[0] == movie):
			return m[1]
	return 0

def select_movie_currently_watching():
    all_movies = getAllMovies()
    movies_list = [movie[0] for movie in all_movies]
    option = st.selectbox("Movies!", movies_list)
    return get_movie_id_from_list(all_movies, option)
    

def deleteExistingMovie():
	col1, col2= st.columns(2)

	all_db = pd.DataFrame(getAllMovies(), columns=["title", "id"])
	user_movies = view_all_movies(st.session_state.user)
	user_movie_titles = [movie[0] for movie in user_movies]
	new_db =all_db[all_db["title"].apply(lambda x: any(k in x for k in user_movie_titles))]


	new_movie_id = col1.selectbox("Movie", options = new_db["id"], format_func = lambda x: format(new_db, x), key = 3)
	if col2.button("Remove review"):
		st.success("Removed movie from list")
		delete_ratings(st.session_state.user, new_movie_id)

def get_user_movies_association(user_id, items, movies):
  users_movies_association = list()
  excluded_movies = list()

  for item in items:
    if (item[0] == (user_id - 1)):
    	excluded_movies.append(item[1])
  
  for movie in movies:
      users_movies_association.append([user_id-1, movie-1])
  return users_movies_association, excluded_movies


def prepare_user(user_id):
  ratings_data = pd.read_csv("ml-1m/ratings.dat", sep="::", engine='python', header=None).to_numpy()[:, :3]
  movies_data = pd.read_csv("ml-1m/movies.dat", sep="::", engine='python', header=None).to_numpy()[:, :3]
  items = ratings_data[:, :2].astype(np.int) - 1
  items, excluded_movies = get_user_movies_association(user_id, items, movies_data[:, 0])
#   print(excluded_movies)
  return torch.tensor(items).to("cuda"), excluded_movies

def make_prediction_ncf(user_id, k, movies):
	model_test = NCRF([6040,3952], embed_dim=16, mlp_dims=(16, 16), dropout=0.5,
                                            user_idx=[0],
                                            item_idx=[1])
	model_test.load_state_dict(torch.load("model_ncf.bin"))
	fields, excluded_movies = prepare_user(user_id)
	model_test.cuda()
	y = model_test(fields)
	y = y.tolist()
	for index in excluded_movies:
		y.pop(index-1)
	y = np.array(y)
	indices = (-y).argsort()[:k]
	indices = [index + 1 for index in indices]

	return movies.iloc[indices]


def main():
    # prepare_user(1)
	(X, y, dim, movies) = readData()
	# print("asdadddddddddddddddddddddddddddd")
	# prepare_user(1)
	# make_prediction_ncf()
	st.title("Movie Recommender")
	if "user" not in st.session_state:
		# Will store the currently logged user
		st.session_state.user = None

	# run_app_2(X, y, dim, movies, deep=True)

	# The database is already populated. See data.db.
	# populate_databse()
	menu = ["Login","SignUp"]
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == "Login":
		login_func()
	elif choice =="SignUp":
		signup_func()

	if st.session_state.user:
		st.text("Logged in as {}".format(st.session_state.user))
		page = st.selectbox("Choose your model", ["Matrix Factorization Model", "Neural Collaborative Filtering", "LSTM Collaborative Filtering"])

		
		if page == "Matrix Factorization Model":
			print("First Page")
			run_app(st.session_state.user, X, y, dim, movies)

		if page == "Neural Collaborative Filtering":
			print("Second Page")
			run_app_ncf(st.session_state.user, X, y, dim, movies)
		
		if page == "LSTM Collaborative Filtering":
			print("Third Page")
			run_app_lstm(st.session_state.user, X, y, dim, movies)

		movieTable(st.session_state.user)
	else:
		st.warning("Please log in first.")

if __name__ == '__main__':
	main()