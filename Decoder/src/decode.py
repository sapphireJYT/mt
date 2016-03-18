#!/usr/bin/env python

'''                       yatbear
                   sapphirejyt@gmail.com

    HW 2 - Phrase-based Decoding (Encode Reordering)
            JHU 600.468 Machine Translation
     Based on the monotone decoder by Philipp Koehn          '''

import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing French sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)" )
optparser.add_option("-k", "--translation-per-phrase", dest="k", default=10, type="int", help="Limit on number of translations to consider per phrase (default=10)")
optparser.add_option("-s", "--stack-size", dest="s", default=15, type="int", help="Maximum stack size (default=15)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should assign unknown words as-is with probability 1
for word in set(sum(french, ())):
    if(word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    # precompute the cost table for any span of the French sentence
    # cost is the negative log probability
    cost = {}
    # for spans of all possible lengths
    for span in xrange(1, len(f)+1):
        for bop in xrange(len(f)): # begin of the phrase
            eop = bop + span # end of the phrase
            cost[(bop, eop)] = float('inf')
            if f[bop:eop] in tm:
                for phrase in tm[f[bop:eop]]:
                    logprob = phrase.logprob
                    lm_state = tuple()
                    for word in phrase.english.split():
                        lm_state, word_logprob = lm.score(lm_state, word)
                        logprob += word_logprob
                    cost[(bop, eop)] = -logprob

            # go through different phrase combinations
            # find the lowest possible cost for each segment
            for phrase_boundary in xrange(bop+1, eop):
                if (phrase_boundary, eop) not in cost:
                    cost[(phrase_boundary, eop)] = float('inf')
                new_cost = cost[(bop, phrase_boundary)] + cost[(phrase_boundary, eop)]
                cost[(bop, eop)] = min(cost[(bop, eop)], new_cost)

    # compute the cost for the untranslated spans given a coverage vector
    def future_cost_estimate(coverage):
        future_cost = 0.0
        start = -1 # mark the begining of each untranslated span
        # sum the costs over the untranslated spans
        for j, translated in enumerate(coverage):
            if not translated:
                if start == -1:
                    start = j
            else:
                if start > -1:
                    future_cost += cost[(start, j)]
                start = -1
        return future_cost

    initial_coverage = [False] * len(f) # initial coverage
    initial_cost = cost[(0, len(f))]
    hypothesis = namedtuple("hypothesis", "logprob, future_cost, lm_state, predecessor, phrase, coverage")
    initial_hypothesis = hypothesis(0.0, initial_cost, lm.begin(), None, None, initial_coverage)
    # initialize the hypothesis stacks
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]: # prune
            # find all possible untanslated spans of the hypothesis
            untranslated_spans = []
            start = -1  # mark the begining of each untranslated span
            for j, translated in enumerate(h.coverage):
                if not translated:
                    start = j
                else:
                    continue
                for end in xrange(start, len(f)):
                    if h.coverage[end]:
                        break
                    untranslated_spans.append((start, end+1))

            # for each untranslated span in h for which there is a translation
            for start, end in untranslated_spans:
                if f[start:end] in tm:
                    for phrase in tm[f[start:end]]:
                        logprob = h.logprob + phrase.logprob
                        # update the coverage vector
                        new_coverage = h.coverage[:start] + [True]*(end-start) + h.coverage[end:]
                        lm_state = h.lm_state
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            logprob += word_logprob
                        logprob += lm.end(lm_state) if end==len(f) else 0.0
                        # compute new future cost
                        new_future_cost = future_cost_estimate(new_coverage)
                        new_hypothesis = hypothesis(logprob, new_future_cost, lm_state, h, phrase, new_coverage)

                        # recombination
                        s = sum(new_coverage)
                        # partial hypothesis score = logprob - futurecost
                        if lm_state not in stacks[s] or stacks[s][lm_state].logprob - stacks[s][lm_state].future_cost < logprob - new_future_cost:
                            stacks[s][lm_state] = new_hypothesis

    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
