import streamlit as st
import pandas as pd
import numpy as np
import pickle as cPickle
from data_base import *
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

ratings_path = "ml-1m/ratings.dat"
movie_path = "ml-1m/movies.dat"

names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(ratings_path, sep='::', names=names)


movies = pd.read_csv(movie_path, sep='::', header=None, usecols=[1])
movies.columns = ['title']

X = ratings[['user_id', 'movie_id']].values
y = ratings['rating'].values
dim = (np.unique(np.array(X[:,0])).size, np.unique(np.array(X[:,1])).size)

def create_rating_matrix(X, y, dim):
  r = X[:,0]
  c = X[:,1]
  matrix = csr_matrix((y, (r,c)), shape=(dim[0] + 1, dim[0] + 1))
  M = matrix.todense()
  M = M[1:,1:]
  M = np.asarray(M)
  return M

def load_model_and_get_predictions():
  loaded_model = cPickle.load(open("nmf_model.cpickle", 'rb'))
  W = loaded_model.transform(create_rating_matrix(X, y, dim))
  H = loaded_model.components_.T
  P = H.dot(W.T).T
  P[P > 5] = 5.                   
  P[P < 1] = 1.
  return P

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
	st.subheader("Create An Account")
	new_username = st.text_input("User name")
	new_password = st.text_input("Password",type='password')
	confirm_password = st.text_input('Confirm Password',type='password')
	new_gender = st.text_input("Gender")
	new_age = st.text_input("Age")

	if new_password == confirm_password:
		st.success("Valid Password Confirmed")
	else:
		st.warning("Password not the same")

	if st.button("Sign Up"):
		add_userdata(new_username,new_password, new_gender, new_age)
		st.success("Successfully Created an Account")


def login_func():
	st.subheader("Login Into App")
	username = st.sidebar.text_input("Username")
	password = st.sidebar.text_input("Password",type='password')
	if st.sidebar.checkbox("Login"):
		result = login_user(username,password)
		if result:
			st.success("Logged In as {}".format(username))
			st.subheader("My watched movies")
			users_result = view_all_movies(username)
			clean_db = pd.DataFrame(users_result,columns=["name", "rating"])
			st.dataframe(clean_db)
			
			predicted_rating_matrix = load_model_and_get_predictions()
			initial_rating_matrix = create_rating_matrix(X,y,dim)

			st.subheader("Movie recommendetion")
			recommended_movies = make_recommendation_for_an_existing_user(initial_rating_matrix, predicted_rating_matrix, movies, user_idx=int(username[-1]), k = 5)
			clean_db = pd.DataFrame(recommended_movies,columns=["title"])
			st.dataframe(clean_db)
		else:
			st.warning("Incorrect Username/Password")

def main():

	st.title("Movie Recommender")


	menu = ["Login","SignUp"]
	
	# The database is already populated. See data.db.
	# populate_databse()
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == "Login":
		login_func()
	elif choice =="SignUp":
		signup_func()


if __name__ == '__main__':
	main()