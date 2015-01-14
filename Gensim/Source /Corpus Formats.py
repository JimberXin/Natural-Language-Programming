__author__ = 'JimberXin'

from gensim import corpora
# create a toy corpus of 3 documents,
corpus = [
[(0, 1), (1, 1), (2, 1)],
[(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
[(2, 1), (5, 1), (7, 1), (8, 1)],
[(1, 1), (5, 2), (8, 1)],
[(3, 1), (6, 1), (7, 1)],
[(9, 1)],
[(9, 1), (10, 1)],
[(9, 1), (10, 1), (11, 1)],
[(4, 1), (10, 1), (11, 1)]]

# save as Matrix Market format: save only the non-zero elements of the sparse matrix
# 1st col is documents, 2nd col is word ID, 3rd col is term-frequency
corpora.MmCorpus.serialize('/Users/JimberXin/Documents/Github Workingspace/'
                           'Natural-Language-Programming/Gensim/Corpus/Mycorpus.mm', corpus)

# save as Blei's LDA-C format: topic model of corpus
corpora.BleiCorpus.serialize('/Users/JimberXin/Documents/Github Workingspace/'
                             'Natural-Language-Programming/Gensim/Corpus/Mycorpus.lda-c', corpus)

# save as low corpus
corpora.LowCorpus.serialize('/Users/JimberXin/Documents/Github Workingspace/'
                            'Natural-Language-Programming/Gensim/Corpus/Mycorpus.low', corpus)