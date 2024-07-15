from utils import *
import pprint

def naive_bayes():
	percentage_positive_instances_train = 0.0004
	percentage_negative_instances_train = 0.0004

	percentage_positive_instances_test  = 0.0004
	percentage_negative_instances_test  = 0.0004
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))

	with open('vocab.txt','w') as f:
		for word in vocab:
			f.write("%s\n" % word)
	print("Vocabulary (training set):", len(vocab))
	

if __name__=="__main__":
	naive_bayes()
