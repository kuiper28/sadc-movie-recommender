import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_table("ratings.dat", sep="::", names=["user", "item", "rating", "timestamp"])
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

train_file = open("train.dat", "wt")
test_file = open("test.dat", "wt")

for ind in train_dataset.index:
	line = str(train_dataset["user"][ind]) + "::" + str(train_dataset["item"][ind]) + "::" + str(train_dataset["rating"][ind]) + "::" + str(train_dataset["timestamp"][ind])
	train_file.write(line)
	train_file.write("\n")
 
for ind in test_dataset.index:
	line = str(test_dataset["user"][ind]) + "::" + str(test_dataset["item"][ind]) + "::" +str(test_dataset["rating"][ind]) + "::" + str(test_dataset["timestamp"][ind])
	test_file.write(line)
	test_file.write("\n")
 
train_file.close() 
test_file.close() 