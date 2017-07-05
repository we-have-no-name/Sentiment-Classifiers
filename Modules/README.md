# A neural network classifier for mixed sentiment in tweets

gets a stream of tweets and detects 8 sentiment classes in them in real time

## Requirements
### Python 3 with libraries
* tensorflow 1.0.1
* numpy
* tweepy
* nltk
* matplotlib

### Data
#### The Modules folder
a config.json file of shape
```json
{
	"data_path": "",
	
	"consumer_key": "",
	"consumer_secret": "",

	"access_token": "",
	"access_token_secret": ""
}
```
having your preferred data folder in data_path
and your twitter API access data
they can be optained after creating an app in https://apps.twitter.com

#### The Data folder
the *data_path* folder should have the following
1. folder *d200_word_embedding*
    * can be created by extracting the files in the glove file from http://nlp.stanford.edu/data/glove.twitter.27B.zip
    into a folder named *glove.twitter.27B* in the *data_path* folder
    * then running *Word_Embeddding_Glove_Saver().run()* and choosing *embedding dim = 200*
2. folder *DataSet*
    * has the Data set's csv files each row of shape *[link, tweet, label1, label2, label3]*  
	with labels ranging from 0:7
    * then to train a *session*  
        run *Classifier.py* as a script  
3. or folder *Sessions/DefaultSession* having a *checkpoint* of a trained *session*
## Example of usage
see example_usage.py