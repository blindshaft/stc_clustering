#%%
import os
from collections import Counter
import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
#%%
data_path='data/stackoverflow/'
# %%
with open(data_path + 'vocab_withIdx.dic', 'r') as inp_indx, \
            open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
            open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        emb_index = inp_dic.readlines()
        emb_vec = inp_vec.readlines()
        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec):
            word = index_word[index.replace('\n', '')]
            word_vectors[word] = np.array(list((map(float, vec.split()))))

        del emb_index
        del emb_vec
# %%
with open(data_path + 'title_StackOverflow.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        tempo_list=[]
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)
            tempo_list.append(word_sentence)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
# %%
