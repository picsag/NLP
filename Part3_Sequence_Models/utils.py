import string
import re
import os
import nltk

nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Stop words are messy and not that compelling;
# "very" and "not" are considered stop words, but they are obviously expressing sentiment

# The porter stemmer lemmatizes "was" to "wa".  Seriously???

# I'm not sure we want to get into stop words
stopwords_english = stopwords.words('english')

# Also have my doubts about stemming...
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    '''
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    ### START CODE HERE ###
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
    ### END CODE HERE ###
    return tweets_clean


# let's not reuse variables
# all_positive_tweets = twitter_samples.strings('positive_tweets.json')
# all_negative_tweets = twitter_samples.strings('negative_tweets.json')

def load_tweets():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    return all_positive_tweets, all_negative_tweets


# Layers have weights and a foward function.
# They create weights when layer.initialize is called and use them.
# remove this or make it optional

class Layer(object):
    """Base class for layers."""

    def __init__(self):
        self.weights = None

    def forward(self, x):
        raise NotImplementedError

    def init_weights_and_state(self, input_signature, random_key):
        pass

    def init(self, input_signature, random_key):
        self.init_weights_and_state(input_signature, random_key)
        return self.weights

    def __call__(self, x):
        return self.forward(x)


def get_vocab(vocab_path, tags_path):
    """
    Inputs: vocab_path: path to text file with single word per line.
            tags_path: path to tags, where for each word from vocab_path file is a label
            The number of lines of the vocab_path must be equal to no of lines of the tags_path
    Returns: vocab - a dictionary that translates a word string to a unique number.
    The dictionary contains a <PAD> token.  This is for the length of a sentences to a certain number by adding
    the generic <PAD> token to fill all the empty spaces.
    """
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for i, word in enumerate(f.read().splitlines()):
            vocab[word] = i  # to avoid the 0
        # loading tags (we require this to map tags to their indices)
    vocab['<PAD>'] = len(vocab) # 35180
    tag_map = {}
    with open(tags_path, encoding="utf8") as f:
        for i, t in enumerate(f.read().splitlines()):
            tag_map[t] = i 

    # tag_map['UNK'] = len(tag_map) + 1

    return vocab, tag_map


def get_params(vocab, tag_map, sentences_file, labels_file):
    sentences = []
    labels = []

    with open(sentences_file, encoding="utf8") as f:
        for sentence in f.read().splitlines():
            # replace each token by its index if it is in vocab
            # else use index of UNK_WORD
            s = [vocab[token] if token in vocab 
                 else vocab['UNK']
                 for token in sentence.split(' ')]
            sentences.append(s)

    with open(labels_file, encoding="utf8") as f:
        for sentence in f.read().splitlines():
            # replace each label by its index
            l = [tag_map[label] for label in sentence.split(' ')] # I added plus 1 here
            labels.append(l) 
    return sentences, labels, len(sentences)
