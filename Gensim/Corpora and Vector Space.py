__author__ = 'JimberXin'

"""
 @Author:  Junbo Xin
 @Date:  2015/01/11
 @Description: Corpora and Vector Space Tutorials:
               http://radimrehurek.com/gensim/tut1.html
"""

import logging
# set logging events
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)


"""=====================================From String to Vectors=================================="""
from gensim import corpora, models, similarities

# start from documents represented as strings:
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# tokenize the documents, remove common words(using a stoplist) and word appear once
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
token_once = set(word for word in set(all_tokens) if all_tokens.count(word)==1)
texts = [[word for word in text if word not in token_once]
         for text in texts]

"""
for text in texts:
    print text
['human', 'interface', 'computer']
['survey', 'user', 'computer', 'system', 'response', 'time']
['eps', 'user', 'interface', 'system']
['system', 'human', 'system', 'eps']
['user', 'response', 'time']
['trees']
['graph', 'trees']
['graph', 'minors', 'trees']
['graph', 'minors', 'survey']
"""

# Use bag-of-words to convert documents to vectors
# In this document, each document is represented by one vector by (word, frequency)
dictionary = corpora.Dictionary(texts)
"""
# store in the disk
dictionary.save('/Users/JimberXin/Documents/Github Workingspace/'
                'Natural-Language-Programming/Gensim/FirstDict.dict')

print dictionary
Dictionary(12 unique tokens: [u'minors', u'graph', u'system', u'trees', u'eps']...)

# print dictionary.token2id
for item in dictionary.items():
    print item
(11, u'minors')
(10, u'graph')
(5, u'system')
(9, u'trees')
(8, u'eps')
(0, u'computer')
(4, u'survey')
(7, u'user')
(1, u'human')
(6, u'time')
(2, u'interface')
(3, u'response')
 """

# To actually convert tokenized documents to vectors:
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())

"""
print new_vec
[(0,1),(1,1)]
The function doc2bow() simply counts the number of occurences of each distinct word,
converts the word to its integer word id and returns the result as a sparse vector.
The sparse vector [(0, 1), (1, 1)] therefore reads: in the document 'Human computer interaction',
the words computer (id 0) and human (id 1) appear once; the other ten dictionary words appear
(implicitly) zero times.
"""
corpus = [dictionary.doc2bow(text) for text in texts]  # return a [tokenID, tokenCounts] 2-tuples
"""
# store the corpus of MM format in the disk, for later use
corpora.MmCorpus.serialize('/Users/JimberXin/Documents/Github Workingspace/'
                           'Natural-Language-Programming/Gensim/FirstCorpus.mm', corpus)
for each_corpus in corpus:
    print each_corpus
[(0, 1), (1, 1), (2, 1)]
[(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
[(2, 1), (5, 1), (7, 1), (8, 1)]
[(1, 1), (5, 2), (8, 1)]
[(3, 1), (6, 1), (7, 1)]
[(9, 1)]
[(9, 1), (10, 1)]
[(9, 1), (10, 1), (11, 1)]
[(4, 1), (10, 1), (11, 1)]
"""


"""============================Corpus Streaming -- One Document at a time========================"""
class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
"""
print corpus_memory_friendly
<__main__.MyCorpus object at 0x105026a90>

for vector in corpus_memory_friendly:
    print vector
[(0, 1), (1, 1), (2, 1)]
[(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
[(2, 1), (5, 1), (7, 1), (8, 1)]
[(1, 1), (5, 2), (8, 1)]
[(3, 1), (6, 1), (7, 1)]
[(9, 1)]
[(9, 1), (10, 1)]
[(9, 1), (10, 1), (11, 1)]
[(4, 1), (10, 1), (11, 1)]
Although the output is the same as for the plain Python list, the corpus is now much more memory friendly,
because at most one vector resides in RAM at a time. Corpus can now be as large as you want.
"""

# Similarly, to construct the dictionary without loading all texts into memory
# collect statics about all tokens
dictionary_memory_friendly = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary_memory_friendly.token2id[stopword] for stopword in stoplist
            if stopword in dictionary_memory_friendly.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary_memory_friendly.dfs.iteritems() if docfreq==1]
dictionary_memory_friendly.filter_tokens(stop_ids+once_ids)  # remove stop words and words that appear only once
dictionary_memory_friendly.compactify()  # remove gaps in id sequence after words that were removed
"""
print dictionary_memory_friendly
"""

temp = corpora.MmCorpus('/Users/JimberXin/Documents/Github Workingspace '
                        'Natural-Language-Programming/Gensim/FirstDict.dict')
print type(temp)