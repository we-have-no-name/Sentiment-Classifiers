'''
An example of the usage of the TwitterAgent and the ClassifierInterface

Runs the TwitterAgent to get a stream of tweets and the ClassifierInterface to classify them

Requirements:
a config.json file of shape
{
	"data_path": "",
	
	"consumer_key": "",
	"consumer_secret": "",

	"access_token": "",
	"access_token_secret": ""
}

the data_path folder should have the following
1. folder d200_word_embedding
> can be created by extracting the files in the glove file from http://nlp.stanford.edu/data/glove.twitter.27B.zip
into a folder named glove.twitter.27B in the data_path folder
> then running Word_Embeddding_Glove_Saver().run() and choosing embedding dim = 200
2. folder Sessions/DefaultSession having a checkpoint of a trained session
or
3. folder DataSet with the DataSet's csv files each row of shape [link, tweet, label1, label2, label3]
	with labels ranging from 0:7
> then to train a session
	run Classifier.py as a script
'''
from ClassifierInterface import ClassifierInterface, IncomingQueue
from TwitterAgent import TwitterAgent

# needed for this use case only
import threading, time, sys
import numpy as np

def main():
	'''
	Runs two threads, one to receive a tweets stream and another to classify them.
	'''
	ta = TwitterAgent()
	incoming_queue = IncomingQueue()
	c = ClassifierInterface(incoming_queue)
	ready_queue = c.ready_queue
	max_tweets=[30] #use -1 for an unlimited stream
	def stream_task(q):
		# receive a sample of the global tweets stream
		ta.get_sample_tweets_stream(max_tweets=max_tweets, data_handler=q, lang='en', save_to_files=False)

	def classify_task():
		while(True):
##			time.sleep(0.001)
			c.classify_batch()
			# print classified tweets
			while(ready_queue.qsize()!=0): use_ready_queue(ready_queue)

			# stop after the stream is closed and all tweets are classified
			if (not stream_thread.is_alive() and incoming_queue.qsize()==0): break
		
	stream_thread = threading.Thread(target=stream_task, args = (incoming_queue,))
	stream_thread.start()
	classify_thread = threading.Thread(target=classify_task)
	classify_thread.start()
	
	while(classify_thread.is_alive()):
##		print('waiting:{}, ready:{}\n'.format(incoming_queue.qsize(), ready_queue.qsize()))
		
		# stopping the stream from any thread
		if(False):      # use your condition
			max_tweets[0]=0


# An example usage of the classified tweets
sent_map=['Happy','Love','Hopeful','Neutral','Angry','Hopeless','Hate','Sad']
# used to avoid priting emoji in IDLE
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
def use_ready_queue(ready_queue):
	'''
	Prints the classified tweets.
	'''
	tweet = ready_queue.get()
	probs=', '.join(['{}: {:.3}'.format(sent_map[l], tweet.sentiment[l]) for l in np.argsort(tweet.sentiment)[::-1][:3] if tweet.sentiment[l]>0.01])
	print('tweet: {}\nprobs: {}\ncountry: {}\nlocation: {}\nlang: {}\n'.format(tweet.text, probs, tweet.country, tweet.location, tweet.lang).translate(non_bmp_map))
	

if __name__ == '__main__': main()
