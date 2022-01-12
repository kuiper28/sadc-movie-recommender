from config import *
import pands as pd
import numpy as np
from utils import *
import torch
from model import *
import torch.optim as optim
from torch.autograd import Variable


def train(model, nb_users, training_set, device, criterion, optimizer):
	for epoch in range(1,EPOCHS+1):
		train_loss = 0
		s = 0. 
		torch.autograd.set_detect_anomaly(True)
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
		print("Traget: ", target[0].reshape(-1).tolist())
		print("Output 2222: ", output[0].reshape(-1).tolist())
		print('epoch: '+str(epoch)+ 'loss: '+ str(train_loss/s))

def evaluation(model, nb_users, nb_movies, training_set, test_set, device, criterion):
	test_loss = 0
	s = 0.

	for id_user in range(nb_users):
		input = Variable(training_set[id_user,:]).unsqueeze(0) 
		input=input.to(device)
		target = Variable(test_set[id_user,:]).unsqueeze(0)
		target=target.to(device)
		if torch.sum(target.data > 0) > 0:
			s+=1.
			output = model(input)
			output=output.to(device)
			target.require_grad = False
			output[target == 0] = 0
			pred_loss = criterion(output, target)
			mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
			test_loss += np.sqrt(pred_loss.item()*mean_corrector)
	print('test loss: '+ str(test_loss/s))
    
def main():
	movies=pd.read_csv(MOVIES_FILE,sep="::",header=None,engine='python', encoding='latin-1')
	users = pd.read_csv(USERS_FILE, sep = '::', header = None, engine = 'python', encoding = 'latin-1')
	ratings = pd.read_csv(RATINGS_FILE, sep = '::', header = None, engine = 'python', encoding = 'latin-1')

	training_set=pd.read_csv(TRAIN_FILE,delimiter='::')
	training_set=np.array(training_set,dtype='int')

	test_set=pd.read_csv(TEST_FILE,delimiter='::')
	test_set=np.array(test_set,dtype='int')


	nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
	nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

	training_set=convert(training_set, nb_users, nb_movies)
	test_set=convert(test_set, nb_users, nb_movies)

	np.save('training_set_proc.npy',training_set)
	np.save('test_set_proc.npy',test_set)
	training_set = np.load('training_set_proc.npy')
	test_set = np.load('test_set_proc.npy')

	training_set = torch.FloatTensor(training_set)
	test_set = torch.FloatTensor(test_set)


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = RNNModel()
	model=model.to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr=0.05,weight_decay=0.5)

	train(model, nb_users, training_set, device, criterion, optimizer)
	evaluation(model, nb_users, nb_movies, training_set, test_set, device, criterion)
	
 
if __name__ == "__main__":
    main()