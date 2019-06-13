import numpy as np
import ujson as json


with open("hotpot_train_v1.1.json", "r") as fh:
    word_mat = np.array(json.load(fh))
fh.close()

print(word_mat.__len__())

with open('hotpot_train_small.json', 'w') as fhh:
    json.dump(word_mat[0:700], fhh)
fhh.close()

with open("hotpot_dev_distractor_v1.json", "r") as fh:
    dis_mat = np.array(json.load(fh))
fh.close()

print(dis_mat.__len__())

with open('hotpot_dev_distrator_small.json', 'w') as fhh:
    json.dump(dis_mat[0:100], fhh)
fhh.close()

with open("hotpot_dev_fullwiki_v1.json", "r") as fh:
    wiki_mat = np.array(json.load(fh))
fh.close()

print(wiki_mat.__len__())

with open('hotpot_dev_fullwiki_small.json', 'w') as fhh:
    json.dump(wiki_mat[0:100], fhh)
fhh.close()


print(word_mat[0:10])
