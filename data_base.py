import sqlite3
import os
from threading import Lock

conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()
lock= Lock()

def create_usertable():
	lock.acquire(True)
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT, gender TEXT, age NUMERIC)')
	lock.release()


def add_userdata(username, password, gender, age):
	lock.acquire(True)
	c.execute('INSERT INTO userstable(username,password,gender,age) VALUES (?,?,?,?)',(username,password,gender,age))
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