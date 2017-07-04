from NNGraph import NNGraph
from Classifier import Classifier
import os

def main():
	init()
	train(100)

def init():
	global graph, classifier
	name = input("Graph name: ")
	script_path = os.path.realpath(__file__)
	graph = NNGraph(name = name)

	## Graph experiments
	graph.receive_inputs()
##	graph.rnn(cell_type = 'gru')
##	graph.cnn()
##	graph.merge_rnn_cnn(ratio=0.75, train_ratio=False)

	graph.rnn(act_name='relu')
	graph.cnn(conv_params=[[[30, 2]], [[16,2]]], pool_params=[[32,1],[8,1]], dropout_params=[[None,0.3],None,0.5], dual_embedding=False)
	graph.merge_rnn_cnn(ratio=0.75, train_ratio=False)
	
	
	
	
	## =================
	
	graph.training()
	classifier = Classifier(graph, restore_saved_session=False, trace_run=True)
	classifier.accuracy_analysis.script_path = script_path

def train(iters):
	print("Training\n")
	classifier.train(iters, print_stats=True)
	save = input("Save session? (y, n) [y]:")
	if save == '' or save == 'y':
		c.save_session()
		print("Session saved")
	print("Done")

if __name__ == '__main__': main()
