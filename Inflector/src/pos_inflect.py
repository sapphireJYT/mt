#!/usr/bin/env python

'''                 Yating Jing
                   yating@jhu.edu

                  HW 5 - Inflector
             unigram LM incorporating tags
            JHU 600.468 Machine Translation         '''

import argparse
import codecs
import sys
import os
from collections import defaultdict
from itertools import izip

parser = argparse.ArgumentParser(description="Inflect a lemmatized corpus")
parser.add_argument("-t", type=str, default="data/train", help="training data prefix")
parser.add_argument("-d", type=str, default="data/dtest", help="test data prefix")
parser.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
parser.add_argument("-w", type=str, default="form", help="word file suffix")
parser.add_argument("-p", type=str, default="tag", help="part-of-speech tag suffix")
args = parser.parse_args()

# Python sucks at UTF-8
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

def inflections(word, lemma, tag):
    if LEMMAS.has_key((lemma, tag)):
        return sorted(LEMMAS[(lemma, tag)].keys(), lambda x, y: cmp(LEMMAS[(lemma, tag)][y], LEMMAS[(lemma, tag)][x]))
    elif LEMMAS.has_key(lemma):
        return sorted(LEMMAS[lemma].keys(), lambda x, y: cmp(LEMMAS[lemma][y], LEMMAS[lemma][x]))
    return [lemma]

def best_inflection(word, lemma, tag):
    return inflections(word, lemma, tag)[0]

if __name__ == '__main__':
    # Build a simple unigram model on the training data
    LEMMAS = defaultdict(defaultdict)
    if args.t:
        def combine(a, b): return '%s.%s' % (a, b)
        def utf8read(file): return codecs.open(file, 'r', 'utf-8')
        # Build the LEMMAS hash, a two-level dictionary mapping lemmas to inflections to counts
        train_words = utf8read(combine(args.t, args.w))
        train_lemmas = utf8read(combine(args.t, args.l))
        train_tags = utf8read(combine(args.t, args.p))
        for words, lemmas, tags in izip(train_words, train_lemmas, train_tags):
            w = words.rstrip().lower().split()
            l = lemmas.rstrip().lower().split()
            p = tags.rstrip().lower().split()
            for word, lemma, tag in izip(w, l, p):
                LEMMAS[lemma][word] = LEMMAS[lemma].get(word, 0) + 1
                LEMMAS[(lemma, tag)][word] = LEMMAS[(lemma, tag)].get(word, 0) + 1

    # Choose the most common inflection for each word and output them as a sentence
    test_words = utf8read(combine(args.d, args.w))
    test_lemmas = utf8read(combine(args.d, args.l))
    test_tags = utf8read(combine(args.d, args.p))
    for words, lemmas, tags in izip(test_words, test_lemmas, test_tags):
        w = words.rstrip().lower().split()
        l = lemmas.rstrip().lower().split()
        p = tags.rstrip().lower().split()
        print ' '.join([best_inflection(x, y, z) for x, y, z in izip(w, l, p)])
