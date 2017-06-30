import tensorflow as tf
from tensorflow.contrib import rnn

class NNGraph():
	'''
	Neural Network Graph
	Build a TensorFlow graph that can be used as a neural network for classifying text
	'''
	def __init__(self, batch_size=50, num_steps=85, use_default_network = False, **kwargs):
		'''
		Initialize the graph data and optionally create a graph using the default options
		batch_size: the size of the input batch to be fed to the graph
		num_steps: the count of the word keys in each input pattern
		use_default_network: create a graph using the default options
		vocab_size: the size of the embedding vocabulary
		embedding_dim: the dimensionality of the embedding
		classes: the count of the possible classes that result from the classifier
		'''
		self.graph = tf.Graph()
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.vocab_size = kwargs.get('vocab_size', int(1.2e6))
		self.embedding_dim = kwargs.get('embedding_dim', 200)
		self.classes = kwargs.get('classes', 8)
		if use_default_network: self.default_network()

	def receive_inputs(self, drop_out=0.0, internal_embedding=True, dual_embedding=True, embedding2_dim=12, multi_class_targets=True):
		'''
		Add the first layer that receives inputs from the outside
		drop_out: the ratio to be dropped out from the inputs
		internal_embedding: receive word keys instead of embeddings and collect the embeddings from the graph's internal word embedding
		dual_embedding: (only with internal_embedding) add a trainable second embedding to each word
		embedding2_dim: the dimensionality of the second embedding
		multi_class_targets: receive probabilies of each target class instead of the index of one top class
		'''
		self.internal_embedding = internal_embedding
		self.dual_embedding = dual_embedding
		self.multi_class_targets = multi_class_targets
		self.embedding2_dim = embedding2_dim
		self.drop_out = drop_out
		with self.graph.as_default():
			tf.set_random_seed(0)
			self.use_dropout = tf.constant(True)
			if internal_embedding:
				self.embedding = tf.Variable(tf.constant(0, dtype=tf.float32, shape=(self.vocab_size, self.embedding_dim)), trainable=False, name='embedding')
				self.inputs_keys = tf.placeholder(tf.int32, (self.batch_size, self.num_steps))
				if dual_embedding:
					self.inputs_p1 = tf.gather(self.embedding, self.inputs_keys)
					self.embedding2 = tf.Variable(tf.random_uniform((self.vocab_size, self.embedding2_dim), 0.0001, 0.001))
					self.inputs_p2 = tf.gather(self.embedding2, self.inputs_keys)
					self.inputs = tf.concat((self.inputs_p1, self.inputs_p2), axis=2)
				else:
					self.inputs = tf.gather(self.embedding, self.inputs_keys)
			else:
				self.inputs = tf.placeholder(tf.float32, (self.batch_size, self.num_steps, self.embedding_dim))
			self.inputs_d = tf.nn.dropout(self.inputs, 1-self.drop_out)
			self.inputs_c = tf.cond(self.use_dropout, lambda: self.inputs_d, lambda: self.inputs)
			if multi_class_targets:
				self.targets_mc =  tf.constant(0.0, tf.float32, (self.batch_size, self.classes))
			else:
				self.targets = tf.constant(-1, tf.int32, (self.batch_size,))
				self.targets_oh = tf.one_hot(self.targets, self.classes, on_value=1, off_value=0)
			return True

	def rnn(self, num_units=200, num_layers=1, cell_type='gru', gru_act=None):
		'''
		Build a multi layer RNN
		cell_type: 'gru' or 'lstm'
		gru_act: activation function for the gru cell, None: default (tanh)
		'''
		self.num_units = num_units
		self.num_layers = num_layers
		with self.graph.as_default():
			if cell_type == 'lstm':
				self.cell = rnn.BasicLSTMCell(self.num_units)
			elif cell_type == 'gru':
				if gru_act is None: self.cell = rnn.GRUCell(self.num_units)
				else: self.cell = rnn.GRUCell(self.num_units, activation=gru_act)
			if num_layers>1:
				self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers)
			self.all_outputs, self.final_states = tf.nn.dynamic_rnn(self.cell, self.inputs_c, dtype=tf.float32)
			self.outputs = self.all_outputs[:,-1]
			
			self.softmax_w = tf.Variable(tf.random_uniform((self.num_units, self.classes), 0.0001, 0.001))
			self.softmax_b = tf.Variable(tf.random_uniform((self.classes,), 0.0001, 0.001))
			
			self.rnn_logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
			self.rnn_probs = self.probs = tf.nn.softmax(self.rnn_logits)
		return True
		
	def cnn(self, concat_axis=2, **kwargs):
		'''
		Build a multi layer, multi filter CNN
		conv_params: 3D list of shape [layers, filters, filter_params]
			filter_params: [filters, kernel_size, [stride]]
		pool_params: 2D list of shape [layers, pool_params]
			pool_params: [pool_size, strides]
		use None to skip a layer
		'''
		self.conv_params = kwargs.get('conv_params', [[[100, 1], [100, 2], [50, 7]], [[self.embedding_dim, 2]]])
		self.pool_params = kwargs.get('pool_params', [[6, 3], [6, 2]])
		
		with self.graph.as_default():
			cnn = self.inputs_c
			self.conv, self.pool=[], []
			for i in range(max(len(self.conv_params), len(self.pool_params))):
				if len(self.conv_params)-1 >= i and self.conv_params[i] is not None:
					self.conv.append(self.conv_layer(cnn, self.conv_params[i], concat_axis))
					cnn = self.conv[-1]
				else: self.conv.append(None)
				if len(self.pool_params)-1 >= i and self.pool_params[i] is not None:
					self.pool.append(self.pool_layer(cnn, self.pool_params[i]))
					cnn = self.pool[-1]
				else: self.pool.append(None)

			self.flat = tf.reshape(cnn, (self.batch_size, -1))
			self.dense = tf.layers.dense(inputs=self.flat, units=self.classes, activation=tf.nn.relu)
			self.cnn_logits = self.dense
			self.cnn_probs = self.probs = tf.nn.softmax(self.cnn_logits)
		return True

	def conv_layer(self, inputs, filters_params, concat_axis=2):
		'''
		Build a multi filter conv1d layer
		filters_params: 2D list of shape [filters, filter_params]
			filter_params: [filters, kernel_size, stride]
		'''
		filters = []
		for filter_params in filters_params:
			strides = filter_params[2] if len(filter_params)>2 else 1
			f = tf.layers.conv1d(inputs, filter_params[0], filter_params[1], strides)
			if concat_axis==2:
				if len(filters)>=1: f = tf.pad(f, [[0,0], [0, filters[0].shape[1].value-f.shape[1].value], [0,0]])
			filters.append(f)
		return tf.concat(filters, concat_axis)

	def pool_layer(self, inputs, pool_params):
		'''
		Build a pooling layer
		pool_params: [pool_size, strides] or [devisor_for_top_k]
		'''
		if pool_params[0]==-1: pool_params[0]=inputs.shape[1].value
		if len(pool_params)==2:
			pool = tf.layers.max_pooling1d(inputs, pool_params[0], pool_params[1])
		else:
			k = inputs.shape[1].value//pool_params[0]
			pool = self.top_k_pooling(inputs, k)
		return pool
		

	def top_k_pooling(self, inputs, k):
		'''
		Build a top_k pooling layer
		k: the count of the candidates chosen from inputs
		'''
		return tf.stack([tf.gather(inputs[i], tf.nn.top_k(tf.reduce_mean(inputs[i], axis=-1), k, sorted=False).indices) for i in range(inputs.shape[0])])

	def merge_rnn_cnn(self, factor):
		'''
		Merge the network's RNN with its CNN
		factor: a ratio 0.0:1.0 to be used from RNN probabilities where 1-factor is used for the CNN
		'''
		with self.graph.as_default():
			self.probs = factor * self.rnn_probs + (1-factor) * self.cnn_probs
		return True

	def training(self):
		'''
		Add a loss function and an optimizer
		'''
		with self.graph.as_default():
			self.losses=tf.reduce_sum(tf.square(tf.nn.relu(tf.subtract(self.targets_mc, self.probs))))
			self.opt = tf.train.AdamOptimizer()
			self.opt_op = self.opt.minimize(self.losses)
		return True

	def default_network(self):
		'''
		Build a full network using the default options
		'''
		self.receive_inputs()
		# RNN
		self.rnn()

		# CNN
		## CNN1
		self.cnn()
##		## CNN1.0.1
##		self.cnn(conv_params=[[[100, 1], [100, 2], [50, 7]], [[100, 1], [100, 2], [50, 7]]], pool_params=[[2, 2], [3, 3]])
##		## CNN1.0.2
##		self.cnn(conv_params=[[[100, 1], [100, 2], [50, 7]], [[5, 1], [200, 2], [25, 7]]], pool_params=[[6, 3], [6, 2]])
##		## CNN1.1
##		self.cnn(conv_params=[[[100, 1], [100, 2], [100, 7]], [[100, 1], [100, 2], [100, 7]]], pool_params=[[6, 3], [6]], concat_axis=1)
##		## CNN1.2
##		self.cnn(conv_params=[[[100, 3], [100, 4], [100, 5]], [[self.embedding_dim, 7]]], pool_params=[[2, 2], [3, 3]], concat_axis=1)
##		## CNN1.3
##		self.cnn(conv_params=[[[100, 3], [100, 4], [100, 5]], [[self.embedding_dim, 3, 3]], [[self.embedding_dim, 3, 3]]], pool_params=[None, None], concat_axis=1)
##		## CNN1.4
##		self.cnn(conv_params=[[[100, 1], [100, 2], [100, 3], [100, 4], [100, 5], [100, 6], [100, 7]], [[self.embedding_dim, 3, 3]], [[self.embedding_dim, 3, 3]]], pool_params=[[14], None], concat_axis=1)
##		## CNN2~
##		self.cnn(conv_params=[[[200, 2]], [[200, 2]], [[200, 2]], [[200, 2]], [[200, 2]]], pool_params=[[2, 2], [2, 2], [2, 2], [2, 2], [-1, 1]], concat_axis=1)

		# Merge RNN and CNN
		self.merge_rnn_cnn(0.65)
		self.training()
		return True
