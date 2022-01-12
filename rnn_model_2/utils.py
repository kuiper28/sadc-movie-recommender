import numpy as np

def convert(data, nb_users, nb_movies):
	new_data=[]
	for id_users in range(1,nb_users+1):
		id_movies=data[:,1][data[:,0]==id_users]
		id_ratings=data[:,2][data[:,0]==id_users]
		ratings=np.zeros(nb_movies)
		ratings[id_movies-1]=id_ratings
		new_data.append(list(ratings))
	return new_data