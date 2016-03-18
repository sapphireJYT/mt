#!/usr/bin/env python

'''                             yatbear
                         sapphirejyt@gmail.com

          HW 3 - Automatic Evaluation using Binary Classifiers
                    JHU 600.468 Machine Translation
             Based on the simple evaluator by Philipp Koehn             '''

from __future__ import division
from itertools import islice
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import argparse
import math

N = 8  # n-gram number

# compute the n-gram matches, n = 1, 2, ..., N
def ngram_matches(h, ref, N):
    # count n-gram matches for different n
    matches = []
    for n in xrange(1, N + 1):
        r = ref[:]
        match = 0
        for i in range(1, len(h) - n + 1):
            a = h[i : i + n]
            for j in range(len(r) - n + 1):
                b = r[j : j + n]
                if a == b:
                    del r[j : j + n]
                    match += 1
                    break
        matches.append(match)

    # count the precision, recall and F1-measure of diferent n-grams
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    w = 1.0 #/ N  # used to compute the weighted sum of metrics of different n-grams
    features = []
    for (n, match) in enumerate(matches):
        # w = (N - n + 1) / sum(xrange(1, N+1))
        precision += w * match / (1 if len(h) == n else len(h) - n)
        recall += w * match / (1 if len(ref) == n else len(ref) - n)
        denom = precision + recall
        f1 += 0 if denom == 0 else 2 * w * precision * recall / denom
        features += [precision, recall, f1]
    return features
    # return [precision, recall, f1]

# Get the feature vector from the candidate hypothesis and reference
def extract_features(h, ref):
    # get the n-gram matching fearures
    ngram_match_feature = ngram_matches(h, ref, N)
    # get the word_count_feature
    word_count_feature = min(len(ref)/len(h), 1)  # brevity penalty
    return ngram_match_feature + [word_count_feature]

def main():
    parser = argparse.ArgumentParser(description='Evaluate trasnlation hypothesis.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-l', '--labels', default='data/dev.answers',
            help='labels for supervised training')
    opts = parser.parse_args()

    labels = []
    # load all the answers into the label list
    def answers():
        with open(opts.labels) as l:
            for label in l:
                yield int(label.strip())

    for label in islice(answers(), opts.num_sentences):
        labels.append(label)

    # create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

    train_size = len(labels) # number of labels available for training
    h_features = [] # feature vectors of training data

    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        rset = set(ref)
        h1_features = extract_features(h1, ref)
        h2_features = extract_features(h2, ref)
        h_features.append(h1_features + h2_features)

    classifier = LinearRegression()
    # classifier = LogisticRegression()
    # classifier = svm.SVC()
    classifier.fit(h_features[:train_size], labels)
    predictions = classifier.predict(h_features)

    for prediction in predictions:
        print (1 if prediction > 0 else
                (-1 if prediction < 0
                    else 0))

if __name__ == '__main__':
    main()
