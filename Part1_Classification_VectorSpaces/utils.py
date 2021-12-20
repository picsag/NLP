import numpy as np
import matplotlib.pyplot as plt


def process_tweet(tweet:str):
    pass


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def display_words(freqs:dict):
    # select some words to appear in the report. we will assume that each word is unique (i.e. no duplicates)
    keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
            '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
            'song', 'idea', 'power', 'play', 'magnific']

    # list representing our table of word counts.
    # each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]
    data = []

    # loop through our selected words
    for word in keys:

        # initialize positive and negative counts
        pos = 0
        neg = 0

        # retrieve number of positive counts
        if (word, 1) in freqs:
            pos = freqs[(word, 1)]

        # retrieve number of negative counts
        if (word, 0) in freqs:
            neg = freqs[(word, 0)]

        # append the word counts to the table
        data.append([word, pos, neg])

        fig, ax = plt.subplots(figsize=(8, 8))

        # convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
        x = np.log([x[1] + 1 for x in data])

        # do the same for the negative counts
        y = np.log([x[2] + 1 for x in data])

        # Plot a dot for each pair of words
        ax.scatter(x, y)

        # assign axis labels
        plt.xlabel("Log Positive count")
        plt.ylabel("Log Negative count")

        # Add the word as the label at the same position as you added the points just before
        for i in range(0, len(data)):
            ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

        ax.plot([0, 9], [0, 9], color='red')  # Plot the red line that divides the 2 areas.
        plt.show()


