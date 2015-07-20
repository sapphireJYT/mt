#!/usr/bin/env python

'''                     Yating Jing
                       yating@jhu.edu

                      HW 5 - Inflector
     n-gram LM incorporating POS tags and dependency trees
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
parser.add_argument("-n", type=int, default=4, help="n-gram number")
args = parser.parse_args()

# Python sucks at UTF-8
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
n = args.n

def inflections(n_word, lemma, tag, tree):
    history = tuple(n_word[0:n-2])
    word = n_word[n-1]
    if LEMMAS.has_key((lemma, tag, history)):
        key = (lemma, tag, history)
    elif LEMMAS.has_key((lemma, tag, tree, history)):
        key = (lemma, tag, tree, history)
    elif LEMMAS.has_key((lemma, tag, tree)):
        key = (lemma, tag, tree)
    elif LEMMAS.has_key((lemma, tree, history)):
        key = (lemma, tree, history)
    elif LEMMAS.has_key((lemma, tag)):
        key = (lemma, tag)
    elif LEMMAS.has_key((lemma, tree)):
        key = (lemma, tree)
    elif LEMMAS.has_key((lemma, history)):
        key = (lemma, history)
    elif LEMMAS.has_key(lemma):
        key = lemma
    else:
        return [lemma]
    return sorted(LEMMAS[key].keys(), lambda x, y: cmp(LEMMAS[key][y], LEMMAS[key][x]))

def best_inflection(n_word, lemma, tag, tree):
    return inflections(n_word, lemma, tag, tree)[0]

def find_ngrams(input_list):
    for i in range(n):
        input_list.insert(0, '')
    return zip(*[input_list[i:] for i in range(n)])

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
            (count, word0) = (0, [])

            for word, lemma, tag, node in izip(w, l, p, t):
                tree = (node.parent_index(), node.label())

                # accumulate unigram counts
                LEMMAS[lemma][word] = LEMMAS[lemma].get(word, 0) + 1
                LEMMAS[(lemma, tag)][word] = LEMMAS[(lemma, tag)].get(word, 0) + 1
                LEMMAS[(lemma, tree)][word] = LEMMAS[(lemma, tree)].get(word, 0) + 1
                LEMMAS[(lemma, tag, tree)][word] = LEMMAS[(lemma, tag, tree)].get(word, 0) + 1

                if count < n - 1:
                    count += 1
                    word0.append(word)
                    continue

                # accumulate bigram counts
                history = tuple(word0)
                LEMMAS[(lemma, history)][word] = LEMMAS[(lemma, history)].get(word, 0) + 1
                LEMMAS[(lemma, tag, history)][word] = LEMMAS[(lemma, tag, history)].get(word, 0) + 1
                LEMMAS[(lemma, tree, history)][word] = LEMMAS[(lemma, tree, history)].get(word, 0) + 1
                LEMMAS[(lemma, tag, tree, history)][word] = LEMMAS[(lemma, tag, tree, history)].get(word, 0) + 1
                history = word

    # Choose the most common inflection for each word and output them as a sentence
    test_words = utf8read(combine(args.d, args.w))
    test_lemmas = utf8read(combine(args.d, args.l))
    test_tags = utf8read(combine(args.d, args.p))
    test_trees = utf8read(combine(args.d, args.r))

    for words, lemmas, tags, treestr in izip(test_words, test_lemmas, test_tags, test_trees):
        w = find_ngrams(words.rstrip().lower().split())
        l = lemmas.rstrip().lower().split()
        p = tags.rstrip().lower().split()
        t = DepTree(treestr)
        print ' '.join([best_inflection(x, y, pos, (z.parent_index(), z.label())) for x, y, pos, z in izip(w, l, p, t)])
