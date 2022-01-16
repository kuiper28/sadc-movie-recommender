from cgi import test
from re import I
from config import *
import pandas as pd
import numpy as np
from utils import *
import torch
from model import *
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

def train(model, nb_users, nb_movies, training_set, test_set, device, criterion, optimizer):
	
	best_rmse = 100
	for epoch in range(1,EPOCHS+1):
		train_loss = 0
		s = 0. 
		total_rmse = 0
		for id_user in range(nb_users):
			input = Variable(training_set[id_user,:]).unsqueeze(0)
			input = torch.unsqueeze(input, dim=0)
			input=input.to(device)
			target = input.clone() 
			target=target.to(device)
			if torch.sum(target.data > 0) > 0:
				output = model(input)
				output=output.to(device)
				target.require_grad = False
				output[target == 0] = 0
				loss = criterion(output, target)
				loss.backward()
				train_loss += loss.item()
				s+=1
				optimizer.step()
				target_list = ((target.squeeze(1)).squeeze(0)).reshape(-1).tolist()
				target_list = [value for value in target_list if value != 0]
				output_list = ((output.squeeze(1)).squeeze(0)).reshape(-1).tolist()
				output_list = [value for value in output_list if value != 0]
				rmse = np.sqrt(mean_squared_error(target_list, output_list))
				total_rmse += rmse
		print("Total rmse: ", total_rmse)
		print("Rmse value: {:.3f}".format(total_rmse/s))
		print("Traget: ", target[0].reshape(-1).tolist())
		print("Output 2222: ", output[0].reshape(-1).tolist())
		print('epoch: '+str(epoch)+ 'loss: '+ str(train_loss/s))
		eval_rmse = evaluation(model, nb_users, nb_movies, training_set, test_set, device, criterion)
		print("Eval rmse: {:.5f}".format(eval_rmse))
		if (eval_rmse < best_rmse):
			torch.save(model.state_dict(), "rnn_model.bin")
			print("Model saved!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			best_rmse = eval_rmse	
	
def evaluation(model, nb_users, nb_movies, training_set, test_set, device, criterion):
	test_loss = 0
	s = 0.
	total_rmse = 0
	for id_user in range(nb_users):
		input = Variable(training_set[id_user,:]).unsqueeze(0)
		input = torch.unsqueeze(input, dim=0)
		input=input.to(device)
		target = Variable(test_set[id_user,:]).unsqueeze(0)
		target = torch.unsqueeze(target, dim=0)
		target=target.to(device)
		if torch.sum(target.data > 0) > 0:
			s+=1.
			output = model(input)
			output=output.to(device)
			output[target == 0] = 0
			pred_loss = criterion(output, target)
			test_loss += pred_loss.item()
			target_list = ((target.squeeze(1)).squeeze(0)).reshape(-1).tolist()
			target_list = [value for value in target_list if value != 0]
			output_list = ((output.squeeze(1)).squeeze(0)).reshape(-1).tolist()
			output_list = [value for value in output_list if value != 0]
			rmse = np.sqrt(mean_squared_error(target_list, output_list))
			total_rmse += rmse
	print('test loss: '+ str(test_loss/s))
	return total_rmse/s
    
def predict_top_k_movies(model, user_id, dataset, movies, k, device):
    input = Variable(dataset[user_id, :]).unsqueeze(0)
    input = torch.unsqueeze(input, dim=0)
    movies_for_user = dataset[user_id, :].reshape(-1).tolist()
    ignore_movies = [i for i, e in enumerate(movies_for_user) if e != 0]
    model = model.to(device)
    output = model(input.to(device))
    output = output.to(device)
    output_list = ((output.squeeze(1)).squeeze(0)).reshape(-1).tolist()
    output_list = [e for e in output_list if e not in ignore_movies]
    top_k_positions = sorted(range(len(output_list)), key = lambda sub: output_list[sub])[-k:]
    positions = [list(movies[0]).index(value+1) for value in top_k_positions]
    return [movies[1][index] for index in positions]


def get_movies_users_ratings():
	movies=pd.read_csv(MOVIES_FILE,sep="::",header=None,engine='python', encoding='latin-1')
	users = pd.read_csv(USERS_FILE, sep = '::', header = None, engine = 'python', encoding = 'latin-1')
	ratings = pd.read_csv(RATINGS_FILE, sep = '::', header = None, engine = 'python', encoding = 'latin-1')
	return movies, users, ratings

def main():
	movies, users, ratings = get_movies_users_ratings()
	
 
	training_set = pd.read_csv(TRAIN_FILE,delimiter='::')
	training_set = np.array(training_set,dtype='int')
	
	full_dataset = pd.read_csv(TRAIN_FILE,delimiter='::')
	full_dataset = np.array(full_dataset,dtype='int')
	
	test_set=pd.read_csv(TEST_FILE,delimiter='::')
	test_set=np.array(test_set,dtype='int')


	nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
	nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

	training_set = convert(training_set, nb_users, nb_movies)
	full_dataset = convert(full_dataset, nb_users, nb_movies)
	test_set = convert(test_set, nb_users, nb_movies)

	np.save('training_set_proc.npy',training_set)
	np.save('full_set_proc.npy',full_dataset)
	np.save('test_set_proc.npy',test_set)
	training_set = np.load('training_set_proc.npy')
	full_dataset = np.load('full_set_proc.npy')
	test_set = np.load('test_set_proc.npy')

	training_set = torch.FloatTensor(training_set)
	full_dataset = torch.FloatTensor(full_dataset)
	test_set = torch.FloatTensor(test_set)

 
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = RNNModel( nb_movies, MOVIES_EMBEDDING_DIM, HIDDEN_SIZE)
	model=model.to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr=0.05,weight_decay=0.5)

	train(model, nb_users, nb_movies, training_set, test_set, device, criterion, optimizer)
	predict_top_k_movies(model, 1, full_dataset, movies, 3, device)
	
 
if __name__ == "__main__":
    main()