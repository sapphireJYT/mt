#!/usr/bin/env python

'''                         Yating Jing
                           yating@jhu.edu

             HW 4 - Simple Reranker, add 3 word features
                    JHU 600.468 Machine Translation
           Based on the baseline reranker by Philipp Koehn        '''

from __future__ import division
import optparse
import sys
import string

optparser = optparse.OptionParser()
optparser.add_option("-k", "--kbest-list", dest="input", default="data/dev+test.100best", help="100-best dev+test translation lists(default=data/dev+test.100best)")
optparser.add_option("-s", "--dev-src", dest="dev_src", default="data/dev+test.src", help="Source sentences for dev and test(default=data/dev+test.src)")
optparser.add_option("-l", "--lm", dest="lm", default=-1.0, type="float", help="Language model weight(default=-1.0)")
optparser.add_option("-t", "--tm1", dest="tm1", default=-0.65, type="float", help="Translation model p(e|f) weight(default=-0.65)")
optparser.add_option("-x", "--tm2", dest="tm2", default=-0.8, type="float", help="Lexical translation model p_lex(f|e) weight(default=-0.8)")
optparser.add_option("-b", "--bp", dest="bp", default=0.05, type="float", help="Brevity penalty feature weight(default=0.05)")
optparser.add_option("-c", "--word-count", dest="count", default=0.65, type="float", help="Word count feature weight(default=0.65)")
optparser.add_option("-u", "--untranslated", dest="untranslated", default=0.6, type="float", help="Untranslated word count feature weight(default=0.6)")
(opts, _) = optparser.parse_args()

weights = {'p(e)'           : float(opts.lm),
           'p(e|f)'         : float(opts.tm1),
           'p_lex(f|e)'     : float(opts.tm2),
           'bp'             : float(opts.bp),
           'word_count'     : float(opts.count),
           'untranslated'   : float(opts.untranslated)}

# check if a word can decode ASCII (whether it is in English)
def is_ascii(word):
    try:
        word.decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# score the candidate sentences, choose the best one
def score_sentences(all_hyps, src_sents, num_sents):
    for s in xrange(0, num_sents):
        hyps_for_one_sent = all_hyps[s * 100 : s * 100 + 100]
        (src_num, src) = src_sents[s]
        (best_score, best) = (-1e300, '')
        for (hyp_num, hyp, feats) in hyps_for_one_sent:
            score = 0.0
            for feats in feats.strip().split(' '):  # nlp features
                (k, v) = feats.split('=')
                score += weights[k] * float(v)
            bp = min(len(src.strip().split(' ')) / len(hyp.strip().split(' ')), 1.0) # brevity penalty feature
            score += bp * weights['bp']
            word_count = len(hyp.strip().split(' '))
            score += word_count * weights['word_count']
            untranslated = 0    # untranslated word count feature
            for word in hyp.split(' '):
                if is_ascii(word) is False:
                    untranslated += 1
            score += untranslated / len(src.strip().split(' ')) * weights['untranslated']
            if score > best_score:
                (best, best_score) = (hyp, score)
        try:
            sys.stdout.write("%s\n" % best)
        except(Exception):
            sys.exit(1)
            
def main():
    all_hyps = [pair.split(' ||| ') for pair in open(opts.input)]
    src = [pair.split(' ||| ') for pair in open(opts.dev_src)]
    num_sents = len(src)
    score_sentences(all_hyps, src, num_sents)

if __name__ == '__main__':
    main()
