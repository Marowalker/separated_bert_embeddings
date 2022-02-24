import os
from bert_embedding import BertEmbedding
import mxnet as mx
from collections import defaultdict
import pickle
# import argparse


sentence_list = []
with open('all_sentences.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    sentence_list.append(line.strip())

print("Number of sentences: ", len(sentence_list))

# args = parse_args()
# ctx = mx.gpu(args.gpu_id)
ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx)
result = bert(sentence_list)

# print(result[0])
# print(result[1])

emb_dict = defaultdict()

for tup in result:
    toks = tup[0]
    embs = tup[1]
    for i in range(len(toks)):
        emb_dict[toks[i]] = embs[i]

with open('bert_embedding.pkl', 'wb') as f:
    pickle.dump(emb_dict, f)
