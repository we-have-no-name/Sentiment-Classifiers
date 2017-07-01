import numpy as np
class AccuracyAnalysis:
	'''
	Analyze accuracy of the classifier over time
	'''
	def __init__(self, classes=8, data_set=None):
		'''
		Initialize accuracy arrays
		args:
		classes: count of the classes used by the classifier
		data_set: the data_set used to train the classifier (can be replaced later)
		'''
		self.classes = classes
		self.data_set = data_set
		self.train_accs = np.zeros(0)
		self.test_accs = np.zeros(0)
		self.test_accs2 = np.zeros(0)
		self.test_accs3 = np.zeros(0)
		self.max_train_acc, self.max_test_acc, self.max_test_acc2, self.max_test_acc3 = 0, 0, 0, 0
		self.total_iters = 0; self.prev_iters = 0
		self.saved_iters = []

	def add_probs(self, train_probs, test_probs, i, data_set=None):
		'''
		Receive output probabilities for the data_set from the classifier
		args:
		train_probs: train set probabilities
		test_probs: test set probabilities
		i: current iteration index
		data_set: the data_set used to train the classifier 
		'''
		if data_set is not None: self.data_set = data_set
		self.train_probs = train_probs
		self.test_probs = test_probs
		self.max_train = len(self.train_probs)
		self.max_test = len(self.test_probs)
		self.train_targets = self.data_set.sents_sc_np[:self.max_train]
		self.test_targets = self.data_set.sents_sc_np[self.max_train:self.max_train+self.max_test]
		
		if i==1: self.prev_iters = self.total_iters
		self.total_iters = self.prev_iters+i
		self.saved_iters.append(self.total_iters)
		self.accuracies()
		return self.test_acc, self.max_test_acc

	def accuracies(self):
		'''
		Calculate all accuracies for the data set
		'''
		self.trains_acc0, self.train_acc0 = self.acc0(self.train_probs, self.train_targets)
		trains_acc_r, self.trains_acc, self.train_acc = self.acc(self.train_probs, self.train_targets)
		self.max_train_acc = max(self.train_acc, self.max_train_acc)
		np.concatenate([self.train_accs, np.array([self.train_acc])])
		
		self.tests_acc0, self.test_acc0 = self.acc0(self.test_probs, self.test_targets)
		tests_acc_r, self.tests_acc, self.test_acc = self.acc(self.test_probs, self.test_targets)
		self.max_test_acc = max(self.test_acc, self.max_test_acc)
		np.concatenate([self.test_accs, np.array([self.test_acc])])
		self.tests_acc2, self.test_acc2 = self.acc2(self.tests_acc0, self.tests_acc)
		self.max_test_acc2 = max(self.test_acc2, self.max_test_acc2)
		np.concatenate([self.test_accs2, np.array([self.test_acc2])])
		self.tests_acc3, self.test_acc3 = self.acc3(self.test_probs, self.test_targets, self.tests_acc0, tests_acc_r)
		self.max_test_acc3 = max(self.test_acc3, self.max_test_acc3)
		np.concatenate([self.test_accs3, np.array([self.test_acc3])])
		return True
		
	def acc0(self, probs, targets):
		'''
		Checks if the top output class matches the top target class
		args:
		probs: output probabilities from the classifier
		targets: target probabilities
		'''
		accs0 = np.argsort(probs)[:,-1] == np.argsort(targets)[:,-1]
		acc0 = np.mean(accs0)
		return accs0, acc0

	def acc(self, probs, targets):
		'''
		Calculates min(output_prob, target_prob) for each class and each item
		This means how much of the required probability has been reached
		args:
		probs: output probabilities from the classifier
		targets: target probabilities
		'''
		accs_r = np.minimum(probs, targets);
		accs = np.sum(accs_r, 1)
		acc = np.mean(accs)
		return accs_r, accs, acc

	def acc2(self, accs0, accs):
		'''
		Calculates max(acc0, acc) for each item
		This means if the top class is matched, give the full score, else give acc
		args:
		probs: output probabilities from the classifier
		targets: target probabilities
		'''
		accs2 = np.maximum(accs0, accs)
		acc2 = np.mean(accs2)
		return accs2, acc2

	def acc3(self, probs, targets, accs0, accs_r):
		'''
		Give the full class score if prob reaches 50% of target
		then calculates max(acc3, acc) for each item
		then calculates max(acc3, acc0) for each item
		this means if for any class half the required probability is reached, the full class score is given
		otherwise acc is given
		and if the top prob matches the top target, the full item score is given
		args:
		probs: output probabilities from the classifier
		targets: target probabilities
		'''
		accs3 = np.sum(np.maximum(accs_r, np.minimum((probs> 0.5 * targets), targets)), 1)
		accs3 = np.maximum(accs3, accs0)
		acc3 = np.mean(accs3)
		return accs3, acc3

