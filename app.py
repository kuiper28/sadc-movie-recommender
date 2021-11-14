import streamlit as st
import pandas as pd
import numpy as np

from data_base import *

@st.cache 
def populate_databse():
	names = ['user_id', 'gender', 'age', 'ocupation', 'zip-code']
	users = pd.read_csv("ml-1m/users.dat", sep='::', names=names)
	create_usertable()
	for l in users.values:
		add_userdata(str(l[0]), str(l[0]) + "1234", l[2], l[3])



def signup_func():
	st.subheader("Create An Account")
	new_username = st.text_input("User name")
	new_password = st.text_input("Password",type='password')
	new_gender = st.text_input("Gender")
	new_age = st.text_input("Age")
	confirm_password = st.text_input('Confirm Password',type='password')

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

			task = st.selectbox("Task",["Add Posts","Analytics","Manage"])

			if task == "Add Posts":
				st.subheader("Add Posts")

			elif task == "Analytics":
				st.subheader("Analytics")

			elif task == "Manage":
				st.subheader("Manage Blog")
				users_result = view_all_users()
				clean_db = pd.DataFrame(users_result,columns=["Username","Password"])
				st.dataframe(clean_db)
		else:
			st.warning("Incorrect Username/Password")

def main():

	st.title("Movie Recommender")


	menu = ["Login","SignUp"]
	populate_databse()
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == "Login":
		login_func()
	elif choice =="SignUp":
		signup_func()


if __name__ == '__main__':
	main()