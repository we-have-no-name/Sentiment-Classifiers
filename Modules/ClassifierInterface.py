from queue import Queue
import json
from Classifier import Classifier

class Tweet():
	'''
	Stores main data from a json tweet received by the Twitter Streaming API
	'''
	def __init__(self, tweet):
		'''tweet: tweet json string from the Twitter Streaming API'''
		tweet_dict=json.loads(tweet)
		self.error = False
		try:
			self.id = tweet_dict['id']
			self.text = tweet_dict['text']
			self.lang = tweet_dict['lang']
			self.username = tweet_dict['user']['screen_name']
			self.place = tweet_dict['place']
			self.country = None			
			if self.place is not None:
				self.country = tweet_dict['place']['country']
			self.location = tweet_dict['user']['location']
		except KeyError:
			self.error = True
		self.sentiment = None

class PseudoTweet():
	'''
	Stores a pseudo tweet. for using normal text as a Tweet
	'''
	def __init__(self, tweet):
		'''tweet: list of form [tweet_id, tweet_text]'''
		self.id = tweet[0]
		self.text = tweet[1]
		self.sentiment = None
		self.error = False
	
class IncomingQueue(Queue):
	'''
	Receives tweet objects and adds them to a queue for processing
	'''
	def put(self, tweet):
		'''
		Extracts required data from a tweet json string,
		then adds the generated object to a queue
		'''
		if type(tweet) is not list :
			tweet_object = Tweet(tweet)
		## case of a pseudo tweet
		else: tweet_object = PseudoTweet(tweet)
		
		if tweet_object.error: return False
		else: super().put(tweet_object)
		return True
	
class ClassifierInterface():
	'''
	An interface for classifying tweets from a provided queue using a provided classifier
	'''
	def __init__(self, in_queue, classifier=None):
		'''
		classifier: a classifier that can receive a list of tweet texts and return an array of class probabilities
		in_queue: a queue the classifier can get tweets objects from
		'''
		if classifier is None: classifier = Classifier()
		self.classifier = classifier
		self.in_queue = in_queue
		self.ready_queue = Queue()
		
	def classify_batch(self, batch_size=-1):
		'''
		Classifies a batch of tweets from the in_queue and adds them to the ready_queue
		batch_size: the batch to be extracted, use -1 to get all items from the queue
		get the classified tweets from the objects's ready_queue
		'''
		batch = []
		batch_size = min(batch_size, self.in_queue.qsize())
		if batch_size==-1: batch_size=self.in_queue.qsize()
		for i in range(batch_size):
			batch.append(self.in_queue.get())
		texts = [tweet.text[:min(len(tweet.text), self.classifier.num_steps)] for tweet in batch]
		sentiments = self.classifier.predict(texts)
		for i in range(len(batch)):
			batch[i].sentiment = sentiments[i]
			self.ready_queue.put(batch[i])
		return True

