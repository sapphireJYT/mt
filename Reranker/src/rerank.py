#!/usr/bin/env python

'''                         Yating Jing
                           yating@jhu.edu

            HW 4 - Reranker based on multi-class classifier
                    JHU 600.468 Machine Translation
            Based on the baseline reranker by Philipp Koehn        '''

from __future__ import division
import optparse
import sys
import math
import string
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import svm

optparser = optparse.OptionParser()
optparser.add_option("-b", "--best-list", dest="hyp", default="data/train.100best", help="100-best training translation lists(default=data/train.100best)")
optparser.add_option("-s", "--train-src", dest="src", default="data/train.src", help="Source sentences for training(default=data/train.src)")
optparser.add_option("-r", "--train-ref", dest="ref", default="data/train.ref", help="References sentences for training(default=data/train.ref)")
optparser.add_option("-k", "--kbest-list", dest="input", default="data/dev+test.100best", help="100-best dev+test translation lists(default=data/dev+test.100best)")
optparser.add_option("-d", "--dev-src", dest="dev_src", default="data/dev+test.src", help="Source sentences for dev and test(default=data/dev+test.src)")
optparser.add_option("-f", "--dev-ref", dest="dev_ref", default="data/dev.ref", help="References sentences for training(default=data/dev.ref)")
(opts, _) = optparser.parse_args()

# check if a word can decode ASCII (whether it is in English)
def is_ascii(word):
    try:
        word.decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# extract the language probability, translation probability, lexical translation probablity and count of number of words
def extract_features(all_hyps, src_sents, num_sents):
    feat_vec = []
    for s in xrange(0, num_sents):
        features = []
        hyps_for_one_sent = all_hyps[s * 100 : s * 100 + 100]
        (src_num, src) = src_sents[s]
        
        for (hyp_num, hyp, feats) in hyps_for_one_sent:
            nlp_feats = []
           
            for feats in feats.strip().split(' '):  # nlp features
                (k, v) = feats.split('=')
                nlp_feats += [float(v)]
            
            bp = min(len(src.strip().split(' ')) / len(hyp.strip().split(' ')), 1.0) # brevity penalty feature
            word_count = len(hyp.strip().split(' '))
            untranslated = 0    # untranslated word count feature
            
            for word in hyp.split(' '):
                if is_ascii(word) is False:
                    untranslated += 1
            
            untranslated = untranslated / len(src.strip().split(' '))
            features = nlp_feats + [bp] + [word_count] + [untranslated]
       
        feat_vec.append(features)
    
    return feat_vec

################################################################################

# N-gram F1-measure

N = 4  # n-gram number
# compute the n-gram matches, n = 1, 2, ..., N
def word_matches(h, ref):
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
    w = 1.0 / N  # used to compute the weighted sum of metrics of different n-grams
    for (n, match) in enumerate(matches):
        precision += w * match / (1 if len(h) == n else len(h) - n)
        recall += w * match / (1 if len(ref) == n else len(ref) - n)
        denom = precision + recall
        f1 += 0 if denom == 0 else 2 * w * precision * recall / denom
    return f1

################################################################################
'''
# Bleu matches

# compute the modified n-gram matches
def ngram_match(h, ref, n):
    r = ref[:]
    match = 0;
    for i in range(len(h) - n + 1):
        a = h[i : i + n]
        for j in range(len(r) - n + 1):
            b = r[j : j + n]
            if a == b:
                del r[j : j + n]
                match += n
                break
    return match

# compute the logarithm bleu score of candidate hypothesis
def bleu_score (h, ref, N):
    bp = min(1 - len(ref)/len(h), 0) # brevity penalty
    w = 1.0 / N   # uniform weight
    bleu = bp
    for n in xrange(1, N+1):
        # w = (N - n + 1) / sum(xrange(1, N+1)) # weight, asign more weight to smaller n
        p = ngram_match(h, ref, n)  # modified n-gram precision
        log_p = math.log(p) if p > 0 else float("-inf")
        bleu += w * log_p
    return bleu

# compute the extent to which a candidate hypothesis matches the reference
# using simple METEOR to compute the harmonic mean of precision and recall
def word_matches(h, ref):
    alpha = 0.2   # parameter that balances precision and recall
    match = bleu_score(h, ref, 2)
    recall = match / len(ref)
    precision = match / len(h)
    denom = (1-alpha) * recall + alpha * precision  # denominator
    return 0 if denom == 0 else precision * recall / denom
'''
################################################################################
'''
# simple METEOR computes the harmonic mean of precision and recall
def word_matches(h, ref):
    alpha = 0.05   # parameter that balances precision and recall
    match = sum(1 for w in h if w in ref)
    recall = match / len(ref)
    precision = match / len(h)
    denom = (1-alpha) * recall + alpha * precision  # denominator
    return 0 if denom == 0 else precision * recall / denom
'''
################################################################################

# label candidates according to their similarities to the references
def label(all_hyps, ref_sents, num_sents):
    labels = []
    for s in xrange(0, num_sents):
        hyps_for_one_sent = all_hyps[s * 100 : s * 100 + 100]
        ref = ref_sents[s].strip().split(' ')
        (best_hyp_index, max_match) = (0, -1e300)
        for (i, (hyp_num, hyp_sent, feats)) in enumerate(hyps_for_one_sent):
            hyp = hyp_sent.strip().split(' ')
            hyp_match = word_matches(hyp, ref)
            if hyp_match > max_match:
                (best_hyp_index, max_match) = (i, hyp_match)
        labels.append(best_hyp_index)
    return labels

def main():
    all_hyps = [pair.split(' ||| ') for pair in open(opts.input)]
    src_sents = [pair.split(' ||| ') for pair in open(opts.src)]
    ref_sents = [ref.strip() for ref in open(opts.ref)]
    num_sents = len(ref_sents)

    test_hyps = [pair.split(' ||| ') for pair in open(opts.input)]
    test_srcs = [pair.split(' ||| ') for pair in open(opts.dev_src)]
    test_refs = [ref.strip(' ||| ') for ref in open(opts.dev_ref)]
    test_num_sents = len(test_srcs)

    num_sents += int(test_num_sents / 2)
    all_hyps += test_hyps[: num_sents * 100]
    src_sents += test_srcs[: num_sents]
    ref_sents += test_refs[: num_sents]
    
    train_feat_vec = extract_features(all_hyps, src_sents, num_sents)
    test_feat_vec = extract_features(test_hyps, test_srcs, test_num_sents)
    labels = label(all_hyps, ref_sents, num_sents)

    # classifier = OneVsRestClassifier(LinearSVC())
    classifier = LogisticRegression()
    # classifier = svm.SVC()
    classifier.fit(train_feat_vec, labels)
    predictions = classifier.predict(test_feat_vec)

    for s in xrange(num_sents):
        hyps_for_one_sent = test_hyps[s * 100: s * 100 + 100]
        (hyp_num, best, feats) = hyps_for_one_sent[predictions[s]]
        try:
            sys.stdout.write("%s\n" % best)
        except(Exception):
            sys.exit(1)

if __name__ == '__main__':
    main()
