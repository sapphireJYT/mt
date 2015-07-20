#!/usr/bin/env python

'''                     Yating Jing
                       yating@jhu.edu

                      HW 5 - Inflector
     bigram LM incorporating POS tags and dependency trees
               JHU 600.468 Machine Translation                   '''

import argparse
import codecs
import sys
import os
from collections import defaultdict
from itertools import izip
from tree import DepTree

parser = argparse.ArgumentParser(description="Inflect a lemmatized corpus")
parser.add_argument("-t", type=str, default="data/train", help="training data prefix")
parser.add_argument("-d", type=str, default="data/dtest", help="test data prefix")
parser.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
parser.add_argument("-w", type=str, default="form", help="word file suffix")
parser.add_argument("-p", type=str, default="tag", help="part-of-speech tag suffix")
parser.add_argument("-r", type=str, default="tree", help="dependency tree suffix")
args = parser.parse_args()

# Python sucks at UTF-8
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

def inflections(bi_word, lemma, tag, tree):
    (word0, word) = tuple(bi_word)
    if LEMMAS.has_key((lemma, tag, word0)):
        key = (lemma, tag, word0)
    elif LEMMAS.has_key((lemma, tag, tree, word0)):
        key = (lemma, tag, tree, word0)
    elif LEMMAS.has_key((lemma, tag, tree)):
        key = (lemma, tag, tree)
    elif LEMMAS.has_key((lemma, tree, word0)):
        key = (lemma, tree, word0)
    elif LEMMAS.has_key((lemma, tag)):
        key = (lemma, tag)
    elif LEMMAS.has_key((lemma, tree)):
        key = (lemma, tree)
    elif LEMMAS.has_key((lemma, word0)):
        key = (lemma, word0)
    elif LEMMAS.has_key(lemma):
        key = lemma
    else:
        return [lemma]
    return sorted(LEMMAS[key].keys(), lambda x, y: cmp(LEMMAS[key][y], LEMMAS[key][x]))

def best_inflection(bi_word, lemma, tag, tree):
    return inflections(bi_word, lemma, tag, tree)[0]

def find_bigrams(input_list):
    input_list.insert(0, '')
    return zip(input_list, input_list[1:])

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
        train_trees = utf8read(combine(args.t, args.r))

        for words, lemmas, tags, treestr in izip(train_words, train_lemmas, train_tags, train_trees):
            w = words.rstrip().lower().split()
            l = lemmas.rstrip().lower().split()
            p = tags.rstrip().lower().split()
            t = DepTree(treestr)
            (initial, word0) = (True, '')

            for word, lemma, tag, node in izip(w, l, p, t):
                tree = (node.parent_index(), node.label())

                # accumulate unigram counts
                LEMMAS[lemma][word] = LEMMAS[lemma].get(word, 0) + 1
                LEMMAS[(lemma, tag)][word] = LEMMAS[(lemma, tag)].get(word, 0) + 1
                LEMMAS[(lemma, tree)][word] = LEMMAS[(lemma, tree)].get(word, 0) + 1
                LEMMAS[(lemma, tag, tree)][word] = LEMMAS[(lemma, tag, tree)].get(word, 0) + 1

                if initial:
                    initial = False
                    word0 = word
                    continue

                # accumulate bigram counts
                LEMMAS[(lemma, word0)][word] = LEMMAS[(lemma, word0)].get(word, 0) + 1
                LEMMAS[(lemma, tag, word0)][word] = LEMMAS[(lemma, tag, word0)].get(word, 0) + 1
                LEMMAS[(lemma, tree, word0)][word] = LEMMAS[(lemma, tree, word0)].get(word, 0) + 1
                LEMMAS[(lemma, tag, tree, word0)][word] = LEMMAS[(lemma, tag, tree, word0)].get(word, 0) + 1
                word0 = word

    # Choose the most common inflection for each word and output them as a sentence
    test_words = utf8read(combine(args.d, args.w))
    test_lemmas = utf8read(combine(args.d, args.l))
    test_tags = utf8read(combine(args.d, args.p))
    test_trees = utf8read(combine(args.d, args.r))

    for words, lemmas, tags, treestr in izip(test_words, test_lemmas, test_tags, test_trees):
        w = find_bigrams(words.rstrip().lower().split())
        l = lemmas.rstrip().lower().split()
        p = tags.rstrip().lower().split()
        t = DepTree(treestr)
        print ' '.join([best_inflection(x, y, pos, (z.parent_index(), z.label())) for x, y, pos, z in izip(w, l, p, t)])
