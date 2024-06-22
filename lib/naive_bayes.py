import json

import nltk
import numpy as np
from nltk.corpus import twitter_samples

from .tokenizer import process_tweet


def _load_data():
    nltk.download("twitter_samples")
    nltk.download("stopwords")

    # we use all the tweets in the positive and negative datasets as our training data
    train_pos = twitter_samples.strings("positive_tweets.json")
    train_neg = twitter_samples.strings("negative_tweets.json")
    train_x = train_pos + train_neg
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))

    return train_x, train_y


def _build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


def _train_naive_bayes(freqs, train_y):
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]

    D = len(train_y)
    D_pos = np.sum(train_y)
    D_neg = D - D_pos

    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = float(np.log(p_w_pos / p_w_neg))

    return logprior, loglikelihood


def train_naive_bayes() -> bool:
    try:
        train_x, train_y = _load_data()
        freqs = _build_freqs(train_x, train_y)
        logprior, loglikelihood = _train_naive_bayes(freqs, train_y)
        train_result = {"logprior": logprior, "loglikelihood": loglikelihood}

        with open("./pretrained/nb_model.json", "w") as f:
            json.dump(train_result, f, indent=4)
    except Exception as e:
        print(e)


def calculate_naive_bayes(tweet: str) -> float:
    with open("./pretrained/nb_model.json", "r") as f:
        model = json.load(f)
        logprior = model["logprior"]
        loglikelihood = model["loglikelihood"]

    word_l = process_tweet(tweet)
    p = logprior
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]

    return p
