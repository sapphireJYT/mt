#!/usr/bin/env python

'''                   Yating Jing
                     yating@jhu.edu

      HW 3 - Automatic Evaluation using BLEU metric
             JHU 600.468 Machine Translation
      Based on the simple evaluator by Philipp Koehn         '''

from __future__ import division
from itertools import islice
import argparse
import math

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
    # w = 1.0 / N   # uniform weight
    bleu = bp
    for n in xrange(1, N+1):
        w = (N - n + 1) / sum(xrange(1, N+1)) # weight, asign more weight to smaller n
        p = ngram_match(h, ref, n)  # modified n-gram precision
        log_p = math.log(p) if p > 0 else float("-inf")
        bleu += w * log_p
    return bleu

# compute the extent to which a candidate hypothesis matches the reference
# using simple METEOR to compute the harmonic mean of precision and recall
def word_matches(h, ref):
    alpha = 0.8   # parameter that balances precision and recall
    match = bleu_score(h, ref, 4)
    recall = match / len(ref)
    precision = match / len(h)
    denom = (1-alpha) * recall + alpha * precision  # denominator
    return 0 if denom == 0 else precision * recall / denom

def main():
    parser = argparse.ArgumentParser(description='Evaluate trasnlation hypothesis.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    opts = parser.parse_args()

    # create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        h1_match = word_matches(h1, ref)
        h2_match = word_matches(h2, ref)
        print(1 if h1_match > h2_match else
                (0 if h1_match == h2_match
                    else -1))

if __name__ == '__main__':
    main()
