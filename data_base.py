import sqlite3
import os
from threading import Lock

conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()
lock= Lock()

def create_userratings():
	lock.acquire(True)
	c.execute('CREATE TABLE IF NOT EXISTS ratingstable(username TEXT, movieid INTEGER, rating INTEGER)')
	lock.release()

def create_usertable():
	lock.acquire(True)
	c.execute('CREATE TABLE IF NOT EXISTS userstable(userId INTEGER, username TEXT, password TEXT, gender TEXT, age INTEGER)')
	lock.release()


def create_movoietable():
	lock.acquire(True)
	c.execute('CREATE TABLE IF NOT EXISTS movietable(movieid INTEGER, name TEXT, type TEXT)')
	lock.release()

def add_moviedata(movieid, name, type):
	lock.acquire(True)
	c.execute('INSERT INTO movietable(movieid, name, type) VALUES (?,?,?)',(movieid, name, type))
	conn.commit()
	lock.release()

def add_userdata(username, password, gender, age):
	lock.acquire(True)
	c.execute('INSERT INTO userstable(username,password,gender,age) VALUES (?,?,?,?)',(username,password,gender,age))
	conn.commit()
	lock.release()

def add_ratings(username, movieid, rating):
	lock.acquire(True)
	c.execute('INSERT INTO ratingstable(username, movieid, rating) VALUES (?,?,?)',(username, int(movieid), int(rating)))
	conn.commit()
	lock.release()

def login_user(username,password):
	lock.acquire(True)
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	lock.release()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def view_all_movies(username):
	c.execute('SELECT m.name, r.rating from movietable m, ratingstable r where m.movieid=r.movieid and r.username='+ '"'+str(username)+'"')
	data = c.fetchall()
	return data