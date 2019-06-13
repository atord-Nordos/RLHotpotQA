import spacy
from ordered_set import OrderedSet
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import torch
import ujson as json
import bisect
from typing import Dict
import random
import numpy as np
import linecache

# import nltk.data

import time

nlp = spacy.blank("en")
# nlp = nltk.data.load('tokenizers/punkt/english.pickle')
# nlp = English()
# tokenizer = Tokenizer(nlp.vocab)
# tokenizer = English().Defaults.create_tokenizer(nlp)
justatest = []
wordlist = OrderedSet()


# def insert(a, x, lo=0, hi=None):
#     """Basic structure is from the bisect class, insort_left.
#     Insert item x in list a, and keep it sorted assuming a is sorted.
#
#     If x is already in a, insert it to the left of the leftmost x.
#
#     Optional args lo (default 0) and hi (default len(a)) bound the
#     slice of a to be searched.
#     """
#
#     if lo < 0:
#         raise ValueError('lo must be non-negative')
#     if hi is None:
#         hi = len(a)
#     while lo < hi:
#         mid = (lo+hi)//2
#         if a[mid] < x:
#             lo = mid+1
#         else:
#             hi = mid
#     if lo < len(a):
#         if a[lo] == x:
#             return lo       # already in the array
#     a.insert(lo, x)
#     return lo


def token(string):
    words = nlp(string)
    index = []

    for i in words:
        word = str(i)
        index.append(wordlist.add(word))
#         if word not in wordlist:
#         # if len(indexlist) > 1:
#         #     for index in indexlist:
#         #         if len(justatest[index]) == len(word):
#         #             insert(justatest, word, ind, index)
#         #         if len(justatest[index]) > len(word):
#         #             insert(justatest, word, ind, index)
#         #         ind = index
#             newindex = insert(justatest, word)
# #        [x + 1 for x in index if x >= newindex]     # all indices that were after the new index have been moved
#        index.append(newindex)
#             wordlist.add(word)
#
    return index


def embedding(emb_file):
    embedded_list = OrderedSet()
    vec_size = 0
    random.seed(13)

    with open("word_embedding", "w") as f:
        with open(emb_file, "r", encoding="utf-8") as fh:
            for i in fh:
                x = i.split()
                embedded_list.add(x[0])
                if vec_size == 0:
                    vec_size = len(x) - 1
            print(vec_size)

            for s in wordlist:
                if s not in embedded_list:          # If the word does not exist, use a random vector
                    json.dump((np.random.normal(    # because words that were not pre-trained are mostly unique
                        scale=0.01) for _ in range(vec_size)), f)
#                   print(np.random.normal(scale=0.01) for _ in range(vec_size))
                else:
                    json.dump((linecache.getline(emb_file, embedded_list.add(s)).split()[1:]), f)
#                   print(linecache.getline(emb_file, embedded_list.add(s)).split()[1:])
                json.dump('\n', f)          # after a new element was added, add a newline to the json
        fh.close()
    f.close()


def process_file(filename, word_counter=None, char_counter=None):                 # , config, word_counter=None, char_counter=None

    data = json.load(open(filename, 'r'))

    # for k in data:      # the addition of the words to the wordlist
    #     for i in k:
    #         if i == "question":
    #             token(k.get(i))
    #         else:
    #             if i == "context":      # the 'context', includes supporting facts and answers
    #                 for e in k.get(i)[:]:
    #                     token(e[0])     # names of the article of the context
    #                     for s in e[1][:]:
    #                         token(s)    # the context of that respective article

    token("yes")
    position, pos = 0, 0
    with open("train_index.json", "w") as fh:
        for k in data:      # for each data-point
            newdict = k     # a copy of the dict (data-point) to save
            for i in k:
                position = 0
                if i == "supporting_facts":
                    for facts in newdict[i]:
                        newdict[i][position][0] = token(facts[0])
#                        for s in newdict[i][position][0]:
#                            print(wordlist[s])
                        position += 1

                else:
                    if i == "question":
                        newdict[i] = token(k.get(i))
                    else:
                        if i == "context":                      # the 'context'
                            for e in k.get(i)[:]:
                                newdict[i][position][0] = token(e[0])     # names of the article of the context
                                for s in e[1][:]:
                                    newdict[i][position][1][pos] = token(s)    # the context of that respective article
                                    pos += 1
                                position += 1
                                pos = 0
                        else:
                            if i == 'answer':
                                newdict[i] = token(k.get(i))
            json.dump(newdict, fh)
    fh.close()

    embedding("glove.840B.300d.txt")

    # eval_examples = {}
    # tokens = tokenizer(u"This is a sentence")







    # outputs = Parallel(n_jobs=12, verbose=10)(delayed(_process_article)(article, config) for article in data)
    # # outputs = [_process_article(article, config) for article in data]
    # examples = [e[0] for e in outputs]
    # for _, e in outputs:
    #     if e is not None:
    #         eval_examples[e['id']] = e
    #
    # # only count during training
    # if word_counter is not None and char_counter is not None:
    #     for example in examples:
    #         for token in example['ques_tokens'] + example['context_tokens']:
    #             word_counter[token] += 1
    #             for char in token:
    #                 char_counter[char] += 1
    #
    # random.shuffle(examples)
    # print("{} questions in total".format(len(examples)))
    #
    # return examples, eval_examples



if __name__ == '__main__':
    start = time.time()
    process_file("hotpot_train_small.json")  # if it is used as the main, call the test function
    end = time.time()
    print(end - start)