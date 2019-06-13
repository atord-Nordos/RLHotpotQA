import spacy
from ordered_set import OrderedSet
import ujson as json


nlp = spacy.blank("en")

wordlist = OrderedSet()
numbers = [["yes", 1]]


def increase(index):
    numbers[index][1] += 1


def token(string):
    words = nlp(string)

    for i in words:
        word = str(i)
        if word in wordlist:
            increase(wordlist.add(word))
        else:
            wordlist.add(word)
            numbers.append([word, 1])


def process_file(files):
    for filename in files:
        data = json.load(open(filename, 'r'))
        print(filename)
        for k in data:  # for each data-point
            for i in k:
                if i == "supporting_facts":
                    for facts in k.get(i):
                        token(facts[0])
                else:
                    if i == "question":
                        token(k.get(i))
                    else:
                        if i == "context":  # the 'context'
                            for e in k.get(i)[:]:
                                token(e[0])  # names of the article of the context
                                for s in e[1][:]:
                                    token(s)  # the context of that respective article
                        else:
                            if i == 'answer':
                                token(k.get(i))

    with open("wordlist_numbers.json", "w") as fh:
        json.dump(numbers, fh)


if __name__ == '__main__':
    process_file(["hotpot_dev_distractor_v1.json", "hotpot_dev_fullwiki_v1.json", "hotpot_train_v1.1.json"])
