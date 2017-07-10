import tensorflow as tf
from tensorflow.contrib import rnn
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
		name: the name given to the graph
		'''
		self.graph = tf.Graph()
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.vocab_size = kwargs.pop('vocab_size', int(1.2e6))
		self.embedding_dim = kwargs.pop('embedding_dim', 200)
		self.classes = kwargs.pop('classes', 8)
		self.name = kwargs.pop('name', None)
		if kwargs:
			raise TypeError("'{}' is an invalid keyword argument for this function".format(next(iter(kwargs))))
		self.use_default_network = use_default_network
		if self.use_default_network: self.default_network()
		self.set_graph_description()

	def receive_inputs(self, internal_embedding=True, embedding2_dim=None, drop_out=None, multi_class_targets=True):
		'''
		Add the first layer that receives inputs from the outside
		internal_embedding: receive word keys instead of embeddings and collect the embeddings from the graph's internal word embedding
		embedding2_dim: (only with internal_embedding) the dimensionality of the second embedding for each word [default: classes*1.5]
		drop_out: the drop_out applied to the inputs. (use_drop_out must be set True by the session)
		multi_class_targets: receive probabilities of each target class instead of the index of one top class
		'''
		self.internal_embedding = internal_embedding
		if not internal_embedding and embedding2_dim is not None:
			raise ValueError("'embedding2_dim' is not None, can't use dual embedding with external embedding")
		if embedding2_dim is None and internal_embedding: embedding2_dim = int(self.classes * 1.5)
		self.dual_embedding = embedding2_dim is not None and (embedding2_dim > 0)
		self.embedding2_dim = embedding2_dim
		self.inputs_drop_out = drop_out
		self.multi_class_targets = multi_class_targets
		with self.graph.as_default(), tf.name_scope('receive_inputs'):
			tf.set_random_seed(0)
			self.use_drop_out = tf.constant(False)
			if self.internal_embedding:
				self.embedding = tf.Variable(tf.constant(0, dtype=tf.float32, shape=(self.vocab_size, self.embedding_dim)), trainable=False, name='embedding')
				self.embedding_saver = tf.train.Saver({'embedding': self.embedding})
				self.inputs_keys = tf.placeholder(tf.int32, (self.batch_size, self.num_steps))
				self.inputs = tf.gather(self.embedding, self.inputs_keys)
			else:
				self.inputs = tf.placeholder(tf.float32, (self.batch_size, self.num_steps, self.embedding_dim))
			if self.dual_embedding:
				self.embedding2 = tf.Variable(tf.random_uniform((self.vocab_size, self.embedding2_dim), 0.0001, 0.001))
				self.inputs_e2 = tf.gather(self.embedding2, self.inputs_keys)
				self.inputs_de = tf.concat((self.inputs, self.inputs_e2), axis=2)
			if drop_out is not None:
				self.inputs_d = tf.cond(self.use_drop_out, lambda:tf.nn.dropout(self.inputs, 1-self.inputs_drop_out), lambda: self.inputs)
				if self.dual_embedding:
					self.inputs_de_d = tf.cond(self.use_drop_out, lambda:tf.nn.dropout(self.inputs_de, 1-self.inputs_drop_out), lambda: self.inputs_de)
			else:
				self.inputs_d = self.inputs
				if self.dual_embedding:
					self.inputs_de_d = self.inputs_de
			if self.multi_class_targets:
				self.targets_mc =  tf.placeholder(tf.float32, (self.batch_size, self.classes))
			else:
				self.targets = tf.placeholder(tf.int32, (self.batch_size,))
				self.targets_oh = tf.one_hot(self.targets, self.classes, on_value=1, off_value=0)
		self.set_graph_description()

	def rnn(self, num_units=200, num_layers=1, drop_outs = None, cell_type='gru', dual_embedding=None, act_name='tanh'):
		'''
		Build a multi layer RNN
		drop_outs: list [input_drop_out, output_drop_out]. (use_drop_out must be set True by the session)
		cell_type: 'gru' or 'lstm'
		act: activation function for the rnn cell, None: default (tanh)
		dual_embedding: use dual embedding from inputs [default: False if inputs have dual embedding]
		'''
		self.rnn_num_units = num_units
		self.rnn_num_layers = num_layers
		self.rnn_drop_outs = drop_outs
		self.rnn_cell_type = cell_type
		if not self.dual_embedding and dual_embedding is True:
			raise ValueError("can't enable 'dual_embedding', inputs don't have a dual embedding")
		if self.dual_embedding and dual_embedding is None:
			dual_embedding = False
		self.rnn_dual_embedding = dual_embedding
		self.rnn_act_name = act_name
		
		if act_name == 'tanh': act = tf.tanh
		elif act_name == 'relu': act = tf.nn.tanh
		else: raise ValueError('undefined act_name')
		if dual_embedding: inputs_d = self.inputs_de_d
		else: inputs_d = self.inputs_d
		with self.graph.as_default(), tf.name_scope('rnn'):
			if cell_type == 'lstm':
				self.cell = rnn.BasicLSTMCell(num_units, activation=act)
			elif cell_type == 'gru':
				self.cell = rnn.GRUCell(num_units, activation=act)
			else: raise ValueError('undefined cell_type')
			if self.rnn_drop_outs is not None:
				self.cell = rnn.DropoutWrapper(self.cell, 1-self.rnn_drop_outs[0], 1-self.rnn_drop_outs[1])
			if num_layers>1:
				self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers)
			self.all_outputs, self.final_states = tf.nn.dynamic_rnn(self.cell, inputs_d, dtype=tf.float32)
			self.outputs = self.all_outputs[:,-1]
			
			self.softmax_w = tf.Variable(tf.random_uniform((num_units, self.classes), 0.0001, 0.001))
			self.softmax_b = tf.Variable(tf.random_uniform((self.classes,), 0.0001, 0.001))
			
			self.rnn_logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
			self.rnn_probs = self.probs = tf.nn.softmax(self.rnn_logits)
		self.set_graph_description()
		
	def cnn(self, concat_axis=2, dual_embedding=None, **kwargs):
		'''
		Build a multi layer, multi filter CNN
		conv_params: 3D list of shape [layers, filters, filter_params]
			filter_params: [filters, kernel_size, [stride]]
		pool_params: 2D list of shape [layers, pool_params]
			pool_params: [pool_size, strides]
		dropout_params: list [[conv1_drop_out, pool1_drop_out], ..., flat_drop_out, dense_drop_out]
			(use_drop_out must be set True by the session)
		use None to skip a layer
		dual_embedding: use dual embedding from inputs [default: True if inputs have dual embedding]
		'''
		self.conv_params = kwargs.pop('conv_params', [[[100, 1], [100, 2], [50, 7]], [[self.embedding_dim, 2]]])
		self.pool_params = kwargs.pop('pool_params', [[6, 3], [6, 2]])
		self.cnn_dropout_params = kwargs.pop('dropout_params', None)
		if kwargs:
			raise TypeError("'{}' is an invalid keyword argument for this function".format(next(iter(kwargs))))
		if not self.dual_embedding and dual_embedding is True:
			raise ValueError("can't enable 'dual_embedding', inputs don't have a dual embedding")
		if self.dual_embedding and dual_embedding is None:
			dual_embedding = True
		self.cnn_dual_embedding = dual_embedding
		
		if dual_embedding: inputs_d = self.inputs_de_d
		else: inputs_d = self.inputs_d
		
		with self.graph.as_default(), tf.name_scope('cnn'):
			cnn = inputs_d #initially
			self.convs, self.pools, self.cnn_drop_outs=[], [], []
			for i in range(max(len(self.conv_params), len(self.pool_params))):
				params = self.conv_params
				if len(params)-1 >= i and params[i] is not None:
					self.convs.append(self.conv_layer(cnn, self.conv_params[i], concat_axis))
					cnn = self.convs[-1]
				else: self.convs.append(None)
				params = self.cnn_dropout_params
				if params is not None and len(params)-1 >= i and params[i] is not None and params[i][0] is not None:
					self.cnn_drop_outs.append([])
					self.cnn_drop_outs[-1].append(tf.cond(self.use_drop_out, lambda:tf.nn.dropout(cnn, 1-params[i][0]), lambda:cnn))
					cnn = self.cnn_drop_outs[-1][0]
				else: self.cnn_drop_outs.append(None)
				params = self.pool_params
				if len(params)-1 >= i and params[i] is not None:
					self.pools.append(self.pool_layer(cnn, self.pool_params[i]))
					cnn = self.pools[-1]
				else: self.pools.append(None)
				params = self.cnn_dropout_params
				if params is not None and len(params)-1 >= i and params[i] is not None and params[i][1] is not None:
					if self.cnn_dropout_params[i][0] is None: self.cnn_drop_outs.append([None])
					self.cnn_drop_outs[-1].append(tf.cond(self.use_drop_out, lambda:tf.nn.dropout(cnn, 1-params[i][1]), lambda:cnn))
					cnn = self.cnn_drop_outs[-1][1]
				else: self.cnn_drop_outs.append(None)

			self.flat = cnn = tf.reshape(cnn, (-1, cnn.shape[1].value * cnn.shape[2].value))
			params = self.cnn_dropout_params
			if params is not None and len(params)-1 >= i and params[i] is not None:
				self.flat_d = cnn = tf.cond(self.use_drop_out, lambda:tf.nn.dropout(cnn, 1-self.cnn_dropout_params[i]), lambda:cnn)
			i+=1
			self.dense = cnn = tf.layers.dense(inputs=self.flat, units=self.classes, activation=tf.nn.relu)
			params = self.cnn_dropout_params
			if params is not None and len(params)-1 >= i and params[i] is not None:
				self.dense_d = cnn = tf.cond(self.use_drop_out, lambda:tf.nn.dropout(cnn, 1-self.cnn_dropout_params[i]), lambda:cnn)
			self.cnn_logits = cnn
			self.cnn_probs = self.probs = tf.nn.softmax(cnn)
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
		if len(filters)>1: layer =  tf.concat(filters, concat_axis)
		else: layer = filters[0]
		return layer

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

	def merge_rnn_cnn(self, ratio, train_ratio=False):
		'''
		Merge the network's RNN with its CNN
		ratio: a ratio 0.0:1.0 to be used from RNN probabilities where 1-ratio is used for the CNN
		train_ratio: sets whether the merge ratio should be trained
		'''
		self.merge_ratio = ratio
		self.train_ratio = train_ratio
		with self.graph.as_default(), tf.name_scope('merge'):
			self.merge_ratio_variable = tf.clip_by_value(tf.Variable(ratio, trainable = train_ratio), 0, 1)
			self.probs = self.merge_ratio_variable * self.rnn_probs + (1-self.merge_ratio_variable) * self.cnn_probs
		self.set_graph_description()

	def training(self, loss_name = None):
		'''
		Add a loss function and an optimizer
		args:
		loss_name: name of the loss function. options: ('sse_r', 'mse_r', 'mse')
			{default: 'sse_r' if multi_class_targets else 'cross_entropy'}
		'sse_r': sum of squares of RELUs error
		'mse_r': mean of squares of RELUs error
		'mse': mean of squares error
		'''
		if self.multi_class_targets:
			if loss_name is None: loss_name = 'sse_r'
			targets = self.targets_mc
		else:
			if loss_name is None: loss_name = 'cross_entropy'
			targets = tf.cast(self.targets_oh, tf.float32)
		with self.graph.as_default(), tf.name_scope('training'):
			if loss_name == 'sse_r':
				self.losses=tf.reduce_sum(tf.square(tf.nn.relu(tf.subtract(targets, self.probs))))
			elif loss_name == 'mse_r':
				self.losses=tf.reduce_mean(tf.square(tf.nn.relu(tf.subtract(targets, self.probs))))
			elif loss_name == 'sse':
				self.losses=tf.reduce_sum(tf.square(tf.subtract(targets, self.probs)))
			elif loss_name == 'mse':
				self.losses=tf.reduce_mean(tf.square(tf.subtract(targets, self.probs)))
			elif loss_name == 'cross_entropy':
				self.losses=-tf.reduce_sum(targets * tf.log(tf.clip_by_value(self.probs, 1e-10, 1.0)))
			else: raise ValueError('undefined loss_name')
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

##		## CNN1A
##		self.cnn(conv_params=[[[30, 2]], [[16,2]]], pool_params=[[32,1],[8,1]], dropout_params=[[None,0.3],None,0.5], dual_embedding=False)
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
			inputs['embedding2_dim'] = self.embedding2_dim
			inputs['drop_out'] = self.inputs_drop_out
			inputs['multi_class_targets'] = self.multi_class_targets
			description['inputs'] = inputs

		if hasattr(self, 'rnn_probs'):
			rnn = dict()
			rnn['num_units'] = self.rnn_num_units
			rnn['num_layers'] = self.rnn_num_layers
			rnn['drop_outs'] = self.rnn_drop_outs
			rnn['cell_type'] = self.rnn_cell_type
			rnn['dual_embedding'] = self.rnn_dual_embedding
			rnn['act'] = self.rnn_act_name
			description['rnn'] = rnn

		if hasattr(self, 'cnn_probs'):
			cnn = dict()
			cnn['conv_params'] = self.conv_params
			cnn['pool_params'] = self.pool_params
			cnn['dropout_params'] = self.cnn_dropout_params
			cnn['dual_embedding'] = self.cnn_dual_embedding
			description['cnn'] = cnn

		if hasattr(self, 'merge_ratio'):
			merge = dict()
			merge['ratio'] = self.merge_ratio
			merge['train_ratio'] = self.train_ratio
			description['merge'] = merge
			
		if hasattr(self, 'loss_name'):
			training = dict()
			training['loss_name'] = self.loss_name
			training['opt_name'] = self.opt_name
			description['training'] = training
		
		self.description = description


