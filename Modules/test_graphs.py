from NNGraph import NNGraph
from Classifier import Classifier

def main():
	init()
	train(100)

def init():
	global graph, classifier
	name = input("Graph name: ")
	graph = NNGraph(name = name)

	## Graph experiments
	graph.receive_inputs()
	graph.cnn()
	graph.rnn()
	graph.merge_rnn_cnn(ratio=0.75, train_ratio=False)
	
	graph.training()
	classifier = Classifier(graph, restore_saved_session=False)

def train(iters):
	print("Training\n")
	classifier.train(iters, print_stats=True)
	save = input("Save session? (y, n) [y]:")
	if save == '' or save == 'y':
		c.save_session()
		print("Session saved")
	print("Done")

if __name__ == '__main__': main()
