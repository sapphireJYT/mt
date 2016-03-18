#!/usr/bin/env python

'''                     yatbear
                 sapphirejyt@gmail.com

     HW 3 - Automatic Evaluation using METEOR metric
            JHU 600.468 Machine Translation
      Based on the simple evaluator by Philipp Koehn         '''

from __future__ import division
from itertools import islice
import argparse

# simple METEOR computes the harmonic mean of precision and recall
def word_matches(h, ref):
    alpha = 0.15   # parameter that balances precision and recall
    match = sum(1 for w in h if w in ref)
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
        rset = set(ref)
        h1_match = word_matches(h1, rset)
        h2_match = word_matches(h2, rset)
        print(1 if h1_match > h2_match else
                (0 if h1_match == h2_match
                    else -1))

if __name__ == '__main__':
    main()
