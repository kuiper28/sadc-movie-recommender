import streamlit as st
import pandas as pd
import numpy as np
import pickle as cPickle

from streamlit.type_util import Key
from data_base import *
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF



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

@st.cache
def getAllMovies():
	return all_movies()

@st.cache
def create_rating_matrix(X, y, dim):
  r = X[:,0]
  c = X[:,1]
  matrix = csr_matrix((y, (r,c)), shape=(dim[0] + 1, dim[0] + 1))
  M = matrix.todense()
  M = M[1:,1:]
  M = np.asarray(M)
  return M

@st.cache
def load_model_and_get_predictions():
  loaded_model = cPickle.load(open("nmf_model.cpickle", 'rb'))
  W = loaded_model.transform(create_rating_matrix(X, y, dim))
  H = loaded_model.components_.T
  P = H.dot(W.T).T
  P[P > 5] = 5.                   
  P[P < 1] = 1.
  return P

@st.cache
def make_recommendation_for_an_existing_user(initial_rating_matrix, predicted_rating_matrix, movies, user_idx, k=5):
	user_ratings = pd.DataFrame(initial_rating_matrix).iloc[user_idx, :]              
	user_prediction = pd.DataFrame(predicted_rating_matrix).iloc[user_idx,:]
	preferred_movies = np.flip(np.argsort(np.array(user_ratings))[-k:])
	recommended_movies = np.flip(np.argsort(np.array(user_prediction))[-k:])
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
	print(ratings.values[:10])

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



def try_login(username, password):
	result = login_user(username,password)
	if result:
		global logged_in
		logged_in = True
		st.session_state.user = username
	else:
		st.warning("bad")
		st.warning("Incorrect Username/Password")

def run_app(username):
	st.success("Logged In as {}".format(username))

	if st.checkbox("Make movie recommendations"):

		predicted_rating_matrix = load_model_and_get_predictions()
		initial_rating_matrix = create_rating_matrix(X,y,dim)

		st.subheader("Movie recommendations")
		recommended_movies = make_recommendation_for_an_existing_user(initial_rating_matrix, predicted_rating_matrix, movies, user_idx=int(username[-1]), k = 20)
		clean_db = pd.DataFrame(recommended_movies,columns=["title"])
		st.table(clean_db)

	st.header("Add movie ratings")
	addNewMovie()

	st.header("Remove movie ratings")
	deleteExistingMovie()

	st.subheader("My watched movies")
	users_result = view_all_movies(username)
	clean_db = pd.DataFrame(users_result,columns=["name", "rating"])
	st.table(clean_db)
	
	

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
		st.success("Added movie in list")

		add_ratings(st.session_state.user, new_movie_id, new_rating)

def deleteExistingMovie():
	col1, col2= st.columns(2)

	new_db = pd.DataFrame(getAllMovies(), columns=["title", "id"])

	
	new_movie_id = col1.selectbox("Movie", options = new_db["id"], format_func = lambda x: format(new_db, x), key = 3)
	if col2.button("Remove review"):
		st.success("Removed movie from list")
		delete_ratings(st.session_state.user, new_movie_id)
def logout():
	global logged_in
	logged_in = False
	st.session_state.user = None

logged_in = False

def main():
	readData()

	st.title("Movie Recommender")
	if "user" not in st.session_state:
		# Will store the currently logged user
		st.session_state.user = None

	# The database is already populated. See data.db.
	# populate_databse()
	menu = ["Login","SignUp"]
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == "Login":
		login_func()
	elif choice =="SignUp":
		signup_func()

	if st.session_state.user and logged_in:
		run_app(st.session_state.user)
	else:
		st.warning("Please log in first.")

if __name__ == '__main__':
	main()