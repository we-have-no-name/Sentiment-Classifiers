"""
preprocess-twitter.py
python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"
Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
* with further edits to fit this model (use ' ᐸtagᐳ ' instead of '<tag>', split hashtags with underscores)
Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
	text = text.group()
	hashtag_body = text[1:]
	if hashtag_body.isupper():
		result = " ᐸhashtagᐳ {} ᐸallcapsᐳ ".format(hashtag_body)
	else:
		result = " ᐸhashtagᐳ " + re.sub(r"(?=[A-Z])", r" ", hashtag_body, flags=FLAGS)
	result = re.sub(r"_", " ", result, flags=FLAGS)
	return result

def allcaps(text):
	text = text.group()
	return text.lower() + " ᐸallcapsᐳ "


def tokenize(text):
	# Different regex parts for smiley faces
	eyes = r"[8:=;]"
	nose = r"['`\-]?"

	# function so code less repetitive
	def re_sub(pattern, repl):
		return re.sub(pattern, repl, text, flags=FLAGS)

	text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ᐸurlᐳ ")
	text = re_sub(r"/"," / ")
	text = re_sub(r"@\w+", " ᐸuserᐳ ")
	text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " ᐸsmileᐳ ")
	text = re_sub(r"{}{}p+".format(eyes, nose), " ᐸlolfaceᐳ ")
	text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " ᐸsadfaceᐳ ")
	text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " ᐸneutralfaceᐳ ")
	text = re_sub(r"<3"," ᐸheartᐳ ")
	text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " ᐸnumberᐳ ")
	text = re_sub(r"#\S+", hashtag)
	text = re_sub(r"([!?.]){2,}", r"\1  ᐸrepeatᐳ ")
	text = re_sub(r"\b(\S*?)(.)\2{3,}\b", r"\1\2\2  ᐸelongᐳ ")

	## -- I just don't understand why the Ruby script adds ᐸallcapsᐳ to everything so I limited the selection.
	# text = re_sub(r"([^a-z0-9()ᐸᐳ'`\-]){2,}", allcaps)
	text = re_sub(r"([A-Z]){2,}", allcaps)

	return text.lower()

def main():
	text = "I'm TESTING alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
	tokens = tokenize(text)
	tokens
	
if __name__ == "__main__": main()
