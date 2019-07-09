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

nlp = spacy.blank("en")     # the tokenizer
# nlp = nltk.data.load('tokenizers/punkt/english.pickle')
# nlp = English()
# tokenizer = Tokenizer(nlp.vocab)
# tokenizer = English().Defaults.create_tokenizer(nlp)
# justatest = []
wordlist = OrderedSet()     # an Ordered Set which will contain all words of the data once


def token(string):
    words = nlp(string)         # tokenizing of the sentence
    index = []                  # an array for the index numbers, which will replace that sentence

    for i in words:             # for all words:
        word = str(i)           # the token will be transformed into a string in oder to make use of Ordered Set
        index.append(wordlist.add(word))    # it gets added to the Ordered Set 'wordlist'
    return index                # the index is returned so that it can be written into the changed dictionary


def embedding(emb_file):
    embedded_list = OrderedSet()    # a list (set) of the words read from the file
    vector_list = list()            # a list of the vectors for the words
    random.seed(13)                 # random seed which was also used in the hotpot implementation

    with open("word_embedding.json", "w") as f:
        with open(emb_file, "r", encoding="utf-8") as fh:
            vec_size = len(fh.readline().split()) - 1   # the length of the vectors, without the word
            for i in fh:        # for all lines in fh
                x = i.split()   # split it into several 'words'
                word = x[0]     # the first entry is the word, the rest should be the vector
                # print(word)
                if word in wordlist:            # if the word is one we look for in our wordlist
                    embedded_list.add(x[0])     # add the word to a new Set
                    vector = list()             # the rest, being the vector
                    if len(x) == (vec_size + 1):    # if the line is a word + vector instead of several words + vector
                        for vec in x[1:]:           # for the rest elements
                            vector.append(float(vec))   # convert the element into float and add it to the list
                        vector_list.append(vector)  # add the resulting list (vector) to another list
                    # print(vector_list)
            # print(vec_size)

            for s in wordlist:              # for all words in the wordlist
                if s not in embedded_list:          # If the word does not exist in the file, use a random vector
                    array = list()                  # this can be done, because those words are mostly unique
                    for i in np.random.normal(scale=0.01, size=vec_size):
                        array.append(np.round(float(i), 7))     # the return of np.random.normal has \n in it
                    # print(array)
                    json.dump(array, f)             # add it into the new file
                else:
                    index = embedded_list.add(s)        # get the index of the word in the file list
                    json.dump(vector_list[index], f)    # dump the vector at that index into the file
                f.write('\n')               # add a linebreak for easier viewing. Can be skipped
        fh.close()
    f.close()


# def readline(emb_file, index):
#     array = list()
#     for i in linecache.getline(emb_file, index).split()[1:]:  # very slow and inefficient....
#         array.append(float(i))
#     # print(array)
#     # print(len(array))
#     return array


# def embedding_old(emb_file):
#     embedded_list = OrderedSet()
#     vec_size = 0
#     random.seed(13)
#
#     with open("word_embedding.json", "w") as f:
#         with open(emb_file, "r", encoding="utf-8") as fh:
#             for i in fh:
#                 x = i.split()
#                 if x[0] in wordlist:
#                     embedded_list.add(x[0])
#                 if vec_size == 0:
#                     vec_size = len(x) - 1
#             print(vec_size)
#
#             for s in wordlist:
#                 if s not in embedded_list:          # If the word does not exist, use a random vector
#                     array = list()
#                     for i in np.random.normal(scale=0.1, size=vec_size):
#                         array.append(np.round(float(i), 7))
#                     print(array)
#                     json.dump(array, f)
#                 else:
#                     index = embedded_list.add(s)
#                     json.dump(readline(emb_file, index + 1), f)
#                 f.write('\n')
#         fh.close()
#     f.close()


# def process_file_old(filename, word_counter=None, char_counter=None):                 # , config, word_counter=None, char_counter=None
#
#     data = json.load(open(filename, 'r'))   # the data from the current data-set file
#
# #   token("yes")            # in the test set, the word 'yes' had not been contained and thus had been hardcoded
#     position, pos = 0, 0    # positions to determinate the current word (position). Very important in the context
#
#     with open("train_index.json", "w") as fh:       # train_index is the same as the data-set, with the words replaced
#         for k in data:      # for each data-point
#             newdict = k     # a copy of the dict (data-point) to save into the new file
#             for i in k:         # for all elements in the data-point
#                 position = 0
#                 if i == "supporting_facts":     # the supporting facts to be replaced
#                     for facts in newdict[i]:    # the first element is the title, the second is the paragraph number
#                         newdict[i][position][0] = token(facts[0])   # the words get tokenized and replaced by an ID
# #                        for s in newdict[i][position][0]:
# #                            print(wordlist[s])
#                         position += 1           # increase the position at which newdict gets changed
#
#                 else:
#                     if i == "question":         # the question is just a single sentence
#                         newdict[i] = token(k.get(i))
#                     else:
#                         if i == "context":                      # the context, first element is a title, second a list
#                             for e in k.get(i)[:]:
#                                 newdict[i][position][0] = token(e[0])     # names of the article (titles) of the context
#                                 for s in e[1][:]:                         # the rest is a list of sentences
#                                     newdict[i][position][1][pos] = token(s)    # the context of that respective article
#                                     pos += 1
#                                 position += 1
#                                 pos = 0
#                         else:
#                             if i == 'answer':                   # the answer is merely a word in most cases
#                                 newdict[i] = token(k.get(i))
#             json.dump(newdict, fh)      # dump the changed dictionaries into a new file named "train_index.json"
#     fh.close()
#
#     embedding("glove.840B.300d.txt")    # generate a word embedding of the current gathered words in wordlist

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


def process_file(filename, word_counter=None, char_counter=None):                 # , config, word_counter=None, char_counter=None

    data = json.load(open(filename, 'r'))   # the data from the current data-set file

#   token("yes")            # in the test set, the word 'yes' had not been contained and thus had been hardcoded
    pos = 0                 # positions to determinate the current word (position). Very important in the context

    with open("train_index.json", "w") as fh:       # train_index is the same as the data-set, with the words replaced
        for k in data:      # for each data-point
            for i in k:         # for all elements in the data-point
                if i == "supporting_facts":     # the supporting facts to be replaced
                    for facts in k[i]:    # the first element is the title, the second is the paragraph number
                        facts[0] = token(facts[0])   # the words get tokenized and replaced by an ID
#                        for s in newdict[i][position][0]:
#                            print(wordlist[s])

                else:
                    if i == "question":         # the question is just a single sentence
                        k[i] = token(k.get(i))
                    else:
                        if i == "context":                      # the context, first element is a title, second a list
                            for e in k.get(i)[:]:
                                e[0] = token(e[0])     # names of the article (titles) of the context
                                for con in e[1][:]:                         # the rest is a list of sentences
                                    e[1][pos] = token(con)    # the context of that respective article
                                    pos += 1
                                pos = 0
                        else:
                            if i == 'answer':                   # the answer is merely a word in most cases
                                k[i] = token(k.get(i))
        json.dump(data, fh)      # dump the changed dictionaries into a new file named "train_index.json"
    fh.close()

    embedding("glove.840B.300d.txt")    # generate a word embedding of the current gathered words in wordlist


if __name__ == '__main__':
    start = time.time()
    process_file("hotpot_train_small.json")  # if it is used as the main, call the test function
    end = time.time()
    print(end - start)
