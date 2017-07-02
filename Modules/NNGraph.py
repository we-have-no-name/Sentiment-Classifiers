import tensorflow as tf
from tensorflow.contrib import rnn

class NNGraph():
	'''
	Neural Network Graph
	Build a TensorFlow graph that can be used as a neural network for classifying text
	'''
	def __init__(self, batch_size=None, num_steps=90, use_default_network = False, **kwargs):
		'''
		Initialize the graph data and optionally create a graph using the default options
		batch_size: the size of the input batch to be fed to the graph, None: variable size
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
		self.name = kwargs.get('name', None)
		self.use_default_network = use_default_network
		if self.use_default_network: self.default_network()
		self.set_graph_description()

	def receive_inputs(self, internal_embedding=True, embedding2_dim=12, multi_class_targets=True):
		'''
		Add the first layer that receives inputs from the outside
		internal_embedding: receive word keys instead of embeddings and collect the embeddings from the graph's internal word embedding
		embedding2_dim: (only with internal_embedding) the dimensionality of the second embedding for each word
		multi_class_targets: receive probabilies of each target class instead of the index of one top class
		'''
		self.internal_embedding = internal_embedding
		self.dual_embedding = embedding2_dim > 0
		self.multi_class_targets = multi_class_targets
		self.embedding2_dim = embedding2_dim
		with self.graph.as_default(), tf.name_scope('receive_inputs'):
			tf.set_random_seed(0)
			self.drop_out = tf.constant(0.0)
			if self.internal_embedding:
				self.embedding = tf.Variable(tf.constant(0, dtype=tf.float32, shape=(self.vocab_size, self.embedding_dim)), trainable=False, name='embedding')
				self.embedding_saver = tf.train.Saver({'embedding': self.embedding})
				self.inputs_keys = tf.placeholder(tf.int32, (self.batch_size, self.num_steps))
				self.inputs = tf.gather(self.embedding, self.inputs_keys)
				if self.dual_embedding:
					self.embedding2 = tf.Variable(tf.random_uniform((self.vocab_size, self.embedding2_dim), 0.0001, 0.001))
					self.inputs_e2 = tf.gather(self.embedding2, self.inputs_keys)
					self.inputs_de = tf.concat((self.inputs, self.inputs_e2), axis=2)
			else:
				self.inputs = tf.placeholder(tf.float32, (self.batch_size, self.num_steps, self.embedding_dim))
			self.inputs_d = tf.nn.dropout(self.inputs, 1-self.drop_out)
			self.inputs_de_d = tf.nn.dropout(self.inputs_de, 1-self.drop_out)
			if self.multi_class_targets:
				self.targets_mc =  tf.placeholder(tf.float32, (self.batch_size, self.classes))
			else:
				self.targets = tf.placeholder(tf.int32, (self.batch_size,))
				self.targets_oh = tf.one_hot(self.targets, self.classes, on_value=1, off_value=0)
		self.set_graph_description()

	def rnn(self, num_units=200, num_layers=1, cell_type='gru', dual_embedding=False, gru_act=None):
		'''
		Build a multi layer RNN
		cell_type: 'gru' or 'lstm'
		gru_act: activation function for the gru cell, None: default (tanh)
		dual_embedding: use dual embedding from inputs
		'''
		self.rnn_num_units = num_units
		self.rnn_num_layers = num_layers
		self.rnn_cell_type = cell_type
		self.rnn_dual_embedding = dual_embedding
		self.rnn_gru_act = gru_act
		
		if self.dual_embedding and dual_embedding: inputs_d = self.inputs_de_d
		else: inputs_d = self.inputs_d
		with self.graph.as_default(), tf.name_scope('rnn'):
			if cell_type == 'lstm':
				self.cell = rnn.BasicLSTMCell(num_units)
			elif cell_type == 'gru':
				if gru_act is None: self.cell = rnn.GRUCell(num_units)
				else: self.cell = rnn.GRUCell(num_units, activation=gru_act)
			if num_layers>1:
				self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers)
			self.all_outputs, self.final_states = tf.nn.dynamic_rnn(self.cell, inputs_d, dtype=tf.float32)
			self.outputs = self.all_outputs[:,-1]
			
			self.softmax_w = tf.Variable(tf.random_uniform((num_units, self.classes), 0.0001, 0.001))
			self.softmax_b = tf.Variable(tf.random_uniform((self.classes,), 0.0001, 0.001))
			
			self.rnn_logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
			self.rnn_probs = self.probs = tf.nn.softmax(self.rnn_logits)
		self.set_graph_description()
		
	def cnn(self, concat_axis=2, dual_embedding=True, **kwargs):
		'''
		Build a multi layer, multi filter CNN
		conv_params: 3D list of shape [layers, filters, filter_params]
			filter_params: [filters, kernel_size, [stride]]
		pool_params: 2D list of shape [layers, pool_params]
			pool_params: [pool_size, strides]
		use None to skip a layer
		dual_embedding: use dual embedding from inputs
		'''
		self.conv_params = kwargs.get('conv_params', [[[100, 1], [100, 2], [50, 7]], [[self.embedding_dim, 2]]])
		self.pool_params = kwargs.get('pool_params', [[6, 3], [6, 2]])
		self.cnn_dual_embedding = dual_embedding
		
		if self.dual_embedding and dual_embedding: inputs_d = self.inputs_de_d
		else: inputs_d = self.inputs_d
		
		with self.graph.as_default(), tf.name_scope('cnn'):
			cnn = inputs_d
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

			self.flat = tf.reshape(cnn, (-1, cnn.shape[1].value * cnn.shape[2].value))
			self.dense = tf.layers.dense(inputs=self.flat, units=self.classes, activation=tf.nn.relu)
			self.cnn_logits = self.dense
			self.cnn_probs = self.probs = tf.nn.softmax(self.cnn_logits)
		self.set_graph_description()

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
		self.merge_factor = factor
		with self.graph.as_default(), tf.name_scope('merge'):
			self.probs = factor * self.rnn_probs + (1-factor) * self.cnn_probs
		self.set_graph_description()

	def training(self, loss_name = 'sse_r'):
		'''
		Add a loss function and an optimizer
		args:
		loss_name: name of the loss function. options: ('sse_r', 'mse_r', 'mse') ['sse_r']
		'sse_r': sum of squares of RELUs error
		'mse_r': mean of squares of RELUs error
		'mse': mean of squares error
		'''
		with self.graph.as_default(), tf.name_scope('training'):
			if loss_name == 'sse_r':
				self.losses=tf.reduce_sum(tf.square(tf.nn.relu(tf.subtract(self.targets_mc, self.probs))))
			elif loss_name == 'mse_r':
				self.losses=tf.reduce_mean(tf.square(tf.nn.relu(tf.subtract(self.targets_mc, self.probs))))
			else:
				loss_name = 'mse'
				self.losses=tf.reduce_mean(tf.square(tf.subtract(self.targets_mc, self.probs)))
			self.loss_name = loss_name
			self.opt = tf.train.AdamOptimizer()
			self.opt_name = 'adam'
			self.opt_op = self.opt.minimize(self.losses)
			self.global_variables_initializer = tf.global_variables_initializer()
			self.train_saver = tf.train.Saver()
		self.set_graph_description()

	def default_network(self):
		'''
		Build a full network using the default options
		'''
		if self.name is None: self.name = 'default'
		self.receive_inputs()
		# RNN
		self.rnn()

		# CNN
		## CNN1
##		self.cnn()
		
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
##		self.merge_rnn_cnn(0.65)
		
		self.training()
		return True

	def set_graph_description(self):
		'''
		Sets a description for the graph according to the provided parameters. it can be used later for analysis
		'''
		description = dict()
		description['name'] = self.name
		
		init = dict()
		init['batch_size'] = self.batch_size
		init['num_steps'] = self.num_steps
		init['vocab_size'] = self.vocab_size
		init['embedding_dim'] = self.embedding_dim
		init['classes'] = self.classes
		init['use_default_network'] = self.use_default_network
		description['init'] = init

		if hasattr(self, 'internal_embedding'):
			inputs = dict()
			inputs['internal_embedding'] = self.internal_embedding
			inputs['dual_embedding'] = self.dual_embedding
			inputs['multi_class_targets'] = self.multi_class_targets
			inputs['embedding2_dim'] = self.embedding2_dim
			description['inputs'] = inputs

		if hasattr(self, 'rnn_probs'):
			rnn = dict()
			rnn['num_units'] = self.rnn_num_units
			rnn['num_layers'] = self.rnn_num_layers
			rnn['cell_type'] = self.rnn_cell_type
			rnn['dual_embedding'] = self.rnn_dual_embedding
			rnn['gru_act'] = self.rnn_gru_act
			description['rnn'] = rnn

		if hasattr(self, 'cnn_probs'):
			cnn = dict()
			cnn['conv_params'] = self.conv_params
			cnn['pool_params'] = self.pool_params
			cnn['dual_embedding'] = self.cnn_dual_embedding
			description['cnn'] = cnn

		if hasattr(self, 'merge_factor'):
			merge = dict()
			merge['factor'] = self.merge_factor
			description['merge'] = merge
			
		if hasattr(self, 'loss_name'):
			training = dict()
			training['loss_name'] = self.loss_name
			training['opt_name'] = self.opt_name
			description['training'] = training
		
		self.description = description

