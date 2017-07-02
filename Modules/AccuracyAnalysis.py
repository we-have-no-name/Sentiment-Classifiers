import numpy as np
import matplotlib.pyplot as plt
import os, json

class AccuracyAnalysis:
	'''
	Analyze accuracy of the classifier over time
	'''
	def __init__(self, train_stats, classes=8, data_set=None):
		'''
		Initialize accuracy arrays
		args:
		train_stats: a TrainStats object
		classes: count of the classes used by the classifier
		data_set: the data_set used to train the classifier (can be replaced later)
		'''
		self.train_stats = train_stats
		self.classes = classes
		self.data_set = data_set
		self.train_accs = np.zeros(0)
		self.test_accs = np.zeros(0)
		self.test_accs2 = np.zeros(0)
		self.test_accs3 = np.zeros(0)
		self.max_train_acc, self.max_test_acc, self.max_test_acc2, self.max_test_acc3 = 0, 0, 0, 0
		self.train_iters = 0
		self.saved_iters = []
		self.sess_save_path = None
		self.note = ''

	def add_probs(self, train_probs, test_probs, data_set=None):
		'''
		Receive output probabilities for the data_set from the classifier
		args:
		train_probs: train set probabilities
		test_probs: test set probabilities
		data_set: the data_set used to train the classifier 
		'''
		if data_set is not None: self.data_set = data_set
		self.train_probs = train_probs
		self.test_probs = test_probs
		self.max_train = len(self.train_probs)
		self.max_test = len(self.test_probs)
		self.train_targets = self.data_set.sents_sc_np[:self.max_train]
		self.test_targets = self.data_set.sents_sc_np[self.max_train:self.max_train+self.max_test]
		self.baseline_pf()

		self.train_iters = self.train_stats.train_iters
		self.saved_iters.append(self.train_iters)
		self.update_accuracies()
		self.update_statistics()
		self.log()
		return self.test_acc, self.max_test_acc

	def update_accuracies(self):
		'''
		Calculate and update all accuracies for the data set
		'''
		self.trains_acc0, self.train_acc0 = self.acc0(self.train_probs, self.train_targets)
		trains_acc_r, self.trains_acc, self.train_acc = self.acc(self.train_probs, self.train_targets)
		self.max_train_acc = max(self.train_acc, self.max_train_acc)
		self.train_accs = np.concatenate([self.train_accs, np.array([self.train_acc])])
		
		self.tests_acc0, self.test_acc0 = self.acc0(self.test_probs, self.test_targets)
		tests_acc_r, self.tests_acc, self.test_acc = self.acc(self.test_probs, self.test_targets)
		self.max_test_acc = max(self.test_acc, self.max_test_acc)
		self.test_accs = np.concatenate([self.test_accs, np.array([self.test_acc])])
		self.tests_acc2, self.test_acc2 = self.acc2(self.tests_acc0, self.tests_acc)
		self.max_test_acc2 = max(self.test_acc2, self.max_test_acc2)
		self.test_accs2 = np.concatenate([self.test_accs2, np.array([self.test_acc2])])
		self.tests_acc3, self.test_acc3 = self.acc3(self.test_probs, self.test_targets, self.tests_acc0, tests_acc_r)
		self.max_test_acc3 = max(self.test_acc3, self.max_test_acc3)
		self.test_accs3 = np.concatenate([self.test_accs3, np.array([self.test_acc3])])
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
		return:
		accs_r: accuracies per example per class
		accs: sum of accuracies per example
		acc: mean of accuracies of all examples
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
		accs0: accs0 from method acc0
		accs: accs from method acc
		return:
		accs2: sum of accuracies per example
		acc2: mean of accuracies of all examples
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
		accs0: accs0 from method acc0
		accs_r: accs_r from method acc
		return:
		accs3: sum of accuracies per example
		acc3: mean of accuracies of all examples
		'''
		accs3 = np.sum(np.maximum(accs_r, np.minimum((probs> 0.5 * targets), targets)), 1)
		accs3 = np.maximum(accs3, accs0)
		acc3 = np.mean(accs3)
		return accs3, acc3

	def update_statistics(self):
		'''
		Update the statistics of accuracy and training
		'''
		iters = 'iterations ({}:{}):\n'.format(self.saved_iters[0], self.saved_iters[-1])
		latest_acc = 'train_acc: {:.3}  \ttest_acc: {:.3}\n'.format(self.train_accs[-1], self.test_accs[-1])
		top_acc = 'top train_acc: {:.3}\ttop test_acc: {:.3} @{}\n'.format(max(self.train_accs), max(self.test_accs), self.saved_iters[np.argmax(self.test_accs)])
		baseline = 'train_baseline: {:.3}\ttest_baseline: {:.3}\n'.format(self.train_baseline, self.test_baseline)
		pass_test = 'passing test examples: {:.3}\n'.format(np.mean(self.tests_acc>0.5))
		test_acc2 = 'test_acc2: {:.3}  \ttop test_acc2: {:.3} @{}\n'.format(self.test_accs2[-1], max(self.test_accs2), self.saved_iters[np.argmax(self.test_accs2)])
		test_acc3 = 'test_acc3: {:.3}  \ttop test_acc3: {:.3} @{}\n'.format(self.test_accs3[-1], max(self.test_accs3), self.saved_iters[np.argmax(self.test_accs3)])
		std_dev = 'standard deviation: {:.3}\n'.format(np.mean(np.std(self.test_probs, 0)))
		ex_time = "execution time is {}".format(self.sec2clock(self.train_stats.train_time))
		self.statistics = (iters + latest_acc + top_acc + baseline + pass_test + test_acc2 + test_acc3 + std_dev + ex_time)
		return self.statistics
	
	def baseline_pf(self):
		'''
		Calculate the dataset baselines
		'''
		top_probs = np.zeros((3, self.classes))
		priority_probs = [np.array([1]), np.array([2/3, 1/3]), np.array([3/6, 2/6, 1/6])]
		def baseline():
			baseline = 0
			for i in range(3): 
				baseline = max(baseline, np.mean(np.sum(np.minimum(selected_sents, top_probs[i]), 1)))
		#     baseline = np.mean(np.sum(np.minimum(selected_sents, top=_probs[np.sum(selected_sents>0, 1)-1]), 1))
			return baseline
		
		selected_sents = self.train_targets
		top_indices = np.argsort(np.sum(selected_sents, 0))[::-1][:3]
		for i in range(3): top_probs[i, top_indices[:i+1]] = priority_probs[i]
		self.train_baseline = baseline()
		selected_sents = self.test_targets
		self.test_baseline = baseline() if self.max_test!=0 else 0

	def set_plot(self):
		'''
		Prepares the plot of the train and test accuracies
		'''
		plt.gca().cla() 
		plt.plot(self.saved_iters, self.train_accs, label="train(bl=" + '{:.3}'.format(self.train_baseline) + ")")
		plt.plot(self.saved_iters, self.test_accs, label="test (bl=" + '{:.3}'.format(self.test_baseline) + ")")
		plt.title('Accuracy at iterations ({}:{})'.format(self.saved_iters[0], self.saved_iters[-1]))
		plt.legend(loc='best') #upper left

	def tweets_with_results(self, group ='wrong', threshold = 0.5):
		'''
		Gets each tweet with its results from the classifier and targets as a string
		'''
		sent_map=['Happy','Love','Hopeful','Neutral','Angry','Hopeless','Hate','Sad']
		text = ''
		for i in range(self.max_test):
			item_acc = np.sum(self.tests_acc[i])/np.sum(self.data_set.sents_sc_np[self.max_train+i])
			condition = True
			if group == 'wrong': condition = item_acc<threshold
			if group == 'correct': condition = item_acc>=threshold
			if condition:
				tweet = self.data_set.tweets[self.max_train+i]
				ress=', '.join(['{}: {:.3}'.format(sent_map[l], self.test_probs[i, l]) for l in np.argsort(self.test_probs[i])[::-1][:3] if self.test_probs[i, l]>0.01])
				targets=', '.join([sent_map[j] for j in self.data_set.sentiments_lists[self.max_train+i]])
				text += tweet + '\n> ' + ress + " >> " + targets + '\n'
		return text

	def log(self):
		'''
		Saves a log about the training accuracies and results and graph description
		'''
		log_folder = os.path.join(self.train_stats.data_path, 'log', "{} - {}".format(self.train_stats.graph_description['name'], self.train_stats.session_init_time))
		if not os.path.exists(log_folder): os.makedirs(log_folder)
		
		plot_file_name = os.path.join(log_folder, "plot")
		self.set_plot()
		plt.savefig(plot_file_name + '.svg');
		plt.savefig(plot_file_name + '.png');
		plt.close()
		
		stats = self.statistics + ('\n\nsession save path:\n{}'.format(self.sess_save_path) if self.sess_save_path is not None else '')
		with open(os.path.join(log_folder, "statistics.txt"), 'w') as log_file: log_file.write(stats)
		
		with open(os.path.join(log_folder, "results-rejected.txt"), 'w', encoding = 'utf-8') as log_file: log_file.write(self.tweets_with_results())
		with open(os.path.join(log_folder, "results-approved.txt"), 'w', encoding = 'utf-8') as log_file: log_file.write(self.tweets_with_results(group = 'correct'))
		if self.note!='':
			with open(os.path.join(log_folder, "note.txt"), 'w') as log_file: log_file.write(self.note)
		with open(os.path.join(log_folder, "graph_description.txt"), 'w') as log_file: log_file.write(json.dumps(self.train_stats.graph_description, indent=4, sort_keys=True))

	def set_note(self):
		'''
		Set a note to be saved with the log
		'''
		self.note = input("Note:\n")
		
	def sec2clock(self, s):
		'''
		Convert seconds to digital clock like format (88:88:88 or 8 seconds)
		args:
		s: seconds
		'''
		m, s = divmod(s, 60)
		h, m = divmod(m, 60)
		hours = '{:2}:'.format(int(h)) if h else ''
		minutes = '{:2}:'.format(int(m)) if m or h else ''
		seconds = '{:2.0f}'.format(s)
		if not hours and not minutes: seconds+= ' seconds'
		return (hours + minutes + seconds).strip()

class TrainStats():
	'''
	Stores information about training from the trainer
	'''
	def __init__(self):
		'''
		Initializes the train stats
		'''
		self.train_iters = 0
		self.train_time = 0
		self.session_init_time = None
