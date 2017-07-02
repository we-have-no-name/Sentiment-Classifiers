import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import csv, json, pickle
from datetime import datetime
import os

class StreamDataStorage():
	'''stores json tweet strings to files'''
	def __init__(self, file_name, add_timestamp=True):
		'''open the files needed for storage'''
		ts=' - ' + datetime.utcnow().strftime('%Y%m%dT%H%M%S') if add_timestamp else ''
		clean_file_name = file_name + ts + '.csv'
		file_name += ts + '.txt'
		if not os.path.exists('Data'): os.makedirs('Data')
		self.stream_data = open('Data/' + file_name, 'w', encoding='utf-8-sig')
		self.stream_data_clean = open('Data/' + clean_file_name, 'w', encoding='utf-8-sig')
		self.sdata_csv_writer = csv.writer(self.stream_data_clean, lineterminator='\n')
		
	def write(self, data):
		'''receives a json tweet string and write it to the files'''
		tweet_json=json.loads(data)
		try:
			link='www.twitter.com/' + tweet_json['user']['screen_name'] + '/status/' + str(tweet_json['id'])
			if 'retweeted_status' in tweet_json:
				tweet = "RT @" + tweet_json['retweeted_status']['user']['screen_name'] + ": " + tweet_json['retweeted_status']['text']
			else: tweet = tweet_json['text']
		except KeyError:
			return False
		row=[link, tweet]
		self.sdata_csv_writer.writerow(row)
		self.stream_data.write(data)
		return True


class StdOutListener(StreamListener):
	'''specifies how tweet json strings from a tweepy stream are handled'''
	def __init__(self, max_tweets=[-1], lang='', storage_agent=None, data_handler = None, data_list=None):
		'''sets the object's data handler'''
		self.count=0
		self.max_tweets=max_tweets
		self.storage_agent=storage_agent
		self.data_handler=data_handler
		self.data_list=data_list
		self.lang=lang

	def on_data(self, data):
		'''accepts a tweet json string and sends it to the available handlers'''
		no_error = True
		if self.lang!='' and json.loads(data).get('lang', 'no_lang')!=self.lang: return True
		if self.storage_agent is not None: no_error=self.storage_agent.write(data)
		try:
			if self.data_handler is not None: no_error=no_error and self.data_handler.put(data)
		except AttributeError as err:
			raise AttributeError("please use a data_handler object that has a method put(data)")
		if no_error is not False:
			if (self.data_list is not None): self.data_list.append(data)
			self.count+=1
			if self.max_tweets[0]!=-1 and self.count>=self.max_tweets[0]:
				del(self.storage_agent)
				return False
		return True

	def on_error(self, status):
		print(status)
	

class TwitterAgent():
	'''
	TwitterAgent is a simple class that works as a wrapper for tweepy
	for dealing with data from twitter and making your own dataset
	functions:
		make_stream_object : make a tweepy streemobject
		get_sample_tweets_stream :  get realtime tweets
		get_tweets_stream_with_keywords : get a stream of tweets having the provided keywords (can have emojis)
		search_for_tweets_with_keywords : search for tweets having the specified keyword(s) one keyword at a time (can't have emojis)
		get_tweets_with_ids : get tweets object using their ids

	'''
	def __init__(self, config_file='config.json'):
		'''initializes authorization data for the twitter API'''
		try:
			with open(config_file) as json_config_file:
				data = json.load(json_config_file)
				self.consumer_key = data['consumer_key']
				self.consumer_secret = data['consumer_secret']
				self.access_token = data['access_token']
				self.access_token_secret = data['access_token_secret']

			self.auth = OAuthHandler(self.consumer_key, self.consumer_secret)
			self.auth.set_access_token(self.access_token, self.access_token_secret)
			self.api = tweepy.API(self.auth)
		except :
			print("invaild data for auth")
	
	def make_stream_object(self, file_name,**kwargs):
		"""initializes a stream and its data handlers."""
		lang=kwargs.get('lang','')
		add_timestamp=kwargs.get('add_timestamp',True)
		max_tweets=kwargs.get('max_tweets',[20])
		save_to_files=kwargs.get('save_to_files',True)
		data_handler=kwargs.get('data_handler',None)
		data_list=kwargs.get('data_list',None)

		storage_agent=None
		if save_to_files:
			storage_agent = StreamDataStorage(file_name, add_timestamp=add_timestamp)
		self.std_listener = StdOutListener(max_tweets, lang, storage_agent, data_handler, data_list)
		stream = Stream(self.auth, self.std_listener)
		return stream
	
	def store_list_of_objects(self, results):
		"""stores tweet objects in files."""
		ts=' - ' + datetime.utcnow().strftime('%Y%m%dT%H%M%S')
		if not os.path.exists("Data"): os.makedirs("Data") 
		with open('Data/results' + ts + '.csv', 'w', encoding='utf-8-sig') as csv_file:
			csv_writer = csv.writer(csv_file, lineterminator='\n')
			for t in results:
				link='www.twitter.com/' + t.user.screen_name + '/status/' + str(t.id)
				if hasattr(t, 'retweeted_status'):
					tweet = "RT @" + t.retweeted_status.user.screen_name + ": " + t.retweeted_status.text
				else: tweet = t.text
				row=[link, tweet]
				csv_writer.writerow(row)
		with open('Data/results_full' + ts + '.pickle', 'wb') as pickle_full_file:
			pickle.dump(results, pickle_full_file)
		with open('Data/results' + ts + '.json', 'w', encoding='utf-8-sig') as json_file:
			results_json=[]
			for i in range(len(results)):
				results_json.append(results[i]._json)
			json.dump(results_json, json_file, ensure_ascii=False)
	
	def get_sample_tweets_stream(self, **kwargs):
		"""
		get a sample from the stream of tweets flowing through Twitter.
		optionally pass a data_handler object with method put(data)
		to stop the stream at any time set max_tweets[0]=0
		"""
		file_name=kwargs.get('file_name', 'sample_stream_data')
		lang=kwargs.get('lang','en')
		add_timestamp=kwargs.get('add_timestamp',True)
		max_tweets=kwargs.get('max_tweets',[20])
		save_to_files=kwargs.get('save_to_files',True)
		data_handler=kwargs.get('data_handler',None)
		data_list=kwargs.get('data_list',None)
		stream = self.make_stream_object(file_name, lang=lang, add_timestamp=add_timestamp, max_tweets=max_tweets, save_to_files=save_to_files, data_handler=data_handler, data_list=data_list)
		stream.sample()
		return data_list		
	
	def get_tweets_stream_with_keywords(self, keywords, **kwargs):
		"""
		get a stream of tweets having the provided keywords (can have emojis)
		optionally pass a data_handler object with method put(data)
		to stop the stream at any time set max_tweets[0]=0
		use 16 or 32 bit codes for unicode (e.g. emoji='\U0001F602')
		Spaces are ANDs, commas are ORs
		pass a data list to append stream data to
		"""		
		file_name=kwargs.get('file_name','stream_data')
		lang=kwargs.get('lang','en')
		add_timestamp=kwargs.get('add_timestamp',True)
		max_tweets=kwargs.get('max_tweets',[20])
		save_to_files=kwargs.get('save_to_files',True)
		data_handler=kwargs.get('data_handler',None)
		data_list=kwargs.get('data_list',None)
		stream = self.make_stream_object(file_name, lang=lang, add_timestamp=add_timestamp, max_tweets=max_tweets, save_to_files=save_to_files, data_handler=data_handler, data_list=data_list)
		stream.filter(track=keywords)
		return data_list
	
	def search_for_tweets_with_keywords(self, keywords, lang='en', num_per_keyword=20, result_type='mixed', save_to_files=True):
		"""
		search for tweets having the provided keyword(s) (can't have emojis)
		result_type: mixed, recent or popular
		"""
		total_keyworded_tweets = []
		for keyword in keywords :
			keyworded_tweets = self.api.search(keyword, count=num_per_keyword, language=[lang], result_type=result_type)
			total_keyworded_tweets.extend(keyworded_tweets)
		if save_to_files: self.store_list_of_objects(total_keyworded_tweets)
		return total_keyworded_tweets
	
	def get_tweets_with_ids(self, ids, batch_size=100, save_to_files=True):
		"""returns full Tweet objects, specified by the ids parameter, max batch_size is 100"""
		tweet_objects = [];
		for i in range(len(ids)//batch_size+bool(len(ids)%batch_size)):
			ids_batch = ids[i*batch_size:min(len(ids), (i+1)*batch_size)]
			tweet_objects += self.api.statuses_lookup(ids_batch)
		if save_to_files: self.store_list_of_objects(tweet_objects)
		return tweet_objects
	
