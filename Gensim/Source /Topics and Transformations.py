__author__ = 'JimberXin'
"""
    @Author:  Junbo Xin
    @Date:  2015/01/14
    @Desciption: Tutorial of topics and transformations, after corpora and vector space.
"""

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)


# ======================================= Transformation interface ========================================
from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load(
    '/Users/JimberXin/Documents/Github Workingspace/'
    'Natural-Language-Programming/Gensim/Dictionary/FirstDict.dict')
corpus = corpora.MmCorpus(
    '/Users/JimberXin/Documents/Github Workingspace/'
    'Natural-Language-Programming/Gensim/Corpus/FirstCorpus.mm')
print corpus
"""
MmCorpus(9 documents, 12 features, 28 non-zero entries)
"""

# step 1 --- initialize a model, here uses TF-IDF model to convert any vector from the old representation
# (bag-of-words integer counts) to the new representation (TF-IDF real-valued weights)
tfidf = models.TfidfModel(corpus)
doc_bow = [(0, 1), (1, 1)]
# step 2 --- use the model to transform vectors
print tfidf[doc_bow]
'''
[(0, 0.7071067811865476), (1, 0.7071067811865476)]
'''
# apply a transformation to a whole corpus
corpus_tfidf = tfidf[corpus]
corpus_tfidf_round = [[(item[0], float('%0.3f' % item[1])) for item in row] for row in corpus_tfidf]
for doc in corpus_tfidf_round:
    print doc
'''
[(0, 0.577), (1, 0.577), (2, 0.577)]
[(0, 0.444), (3, 0.444), (4, 0.444), (5, 0.324), (6, 0.444), (7, 0.324)]
[(2, 0.571), (5, 0.417), (7, 0.417), (8, 0.571)]
[(1, 0.492), (5, 0.718), (8, 0.492)]
[(3, 0.628), (6, 0.628), (7, 0.459)]
[(9, 1.0)]
[(9, 0.707), (10, 0.707)]
[(9, 0.508), (10, 0.508), (11, 0.696)]
[(4, 0.628), (10, 0.459), (11, 0.628)]
'''
"""
Calling model[corpus] only creates a wrapper around the old corpus document stream--
actual calling corpus_transformed = model[corpus], because that would mean storing the
result in main memory, and that contradicts gensim's objective of memory-independence.
"""
# save tfidf as a model in the disk
tfidf.save('/Users/JimberXin/Documents/Github Workingspace/'
           'Natural-Language-Programming/Gensim/Corpus/model.tfidf')

# ==================================== Latent Semantic Indexing, LSI ==============================================
lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus:bow->tfidf->fold-in-lsi
corpus_lsi_round = [[(item[0], float('%0.3f' % item[1])) for item in row] for row in corpus_lsi]
for doc in corpus_lsi_round:  # both bow->tfidf, tfidf-lsi transformations are executed here, on the fly
    print doc
'''
[(0, 0.066), (1, -0.52)]  #  "Human machine interface for lab abc computer applications"
[(0, 0.197), (1, -0.761)] #  "A survey of user opinion of computer system response time"
[(0, 0.09), (1, -0.724)]  #  "The EPS user interface management system"
[(0, 0.076), (1, -0.632)] #  "System and human system engineering testing of EPS"
[(0, 0.102), (1, -0.574)] #  "Relation of user perceived response time to error measurement"
[(0, 0.703), (1, 0.161)]  #  "The generation of random binary unordered trees"
[(0, 0.877), (1, 0.168)]  #  "The intersection graph of paths in trees"
[(0, 0.91), (1, 0.141)]   #  "Graph minors IV Widths of trees and well quasi ordering"
[(0, 0.617), (1, -0.054)] #  "Graph minors A survey"
'''
print lsi.print_topics(2)
'''
topic #0(1.594): 0.703*"trees" + 0.538*"graph" + 0.402*"minors" + 0.187*"survey" + 0.061*"system"
               + 0.060*"time" + 0.060*"response" + 0.058*"user" + 0.049*"computer" + 0.035*"interface"

topic #1(1.476): -0.460*"system" + -0.373*"user" + -0.332*"eps" + -0.328*"interface" + -0.320*"response"
               + -0.320*"time" + -0.293*"computer" + -0.280*"human" + -0.171*"survey" + 0.161*"trees"

# It appears that according to LSI, 'trees', 'graph', 'minors' are all related words and contributed
# most to the direction of the first topic, while the second topic concerns with all words.
# As a result, the first 5 documents are more related to second topic, the last 4 documents are
# more related to the first topic

'''

# Model persistency is achieved with the save() and load() functions:

lsi.save('/Users/JimberXin/Documents/Github Workingspace/'
         'Natural-Language-Programming/Gensim/Corpus/model.lsi')
# mylsi = models.LsiModel.load(
#         '/Users/JimberXin/Documents/Github Workingspace/'
#         'Natural-Language-Programming/Gensim/Corpus/model.lsi')

# ======================================Latent Dirichlet Allocation, LDA================================
""" Almost the same as LSI"""
lda = models.LdaModel(corpus_tfidf, num_topics=2, id2word=dictionary)
corpus_lda = lda[corpus_tfidf]
corpus_lda_round =[[(item[0], float('%0.3f' % item[1])) for item in row] for row in corpus_lda]
for doc in corpus_lda_round:
    print doc
'''
[(0, 0.769), (1, 0.231)]
[(0, 0.269), (1, 0.731)]
[(0, 0.668), (1, 0.332)]
[(0, 0.7), (1, 0.3)]
[(0, 0.26), (1, 0.74)]
[(0, 0.386), (1, 0.614)]
[(0, 0.348), (1, 0.652)]
[(0, 0.303), (1, 0.697)]
[(0, 0.271), (1, 0.729)]
'''
print lda.print_topics(2)
'''
topic #0 (0.500): 0.139*trees + 0.101*minors + 0.098*graph + 0.086*interface + 0.085*human
                + 0.082*computer + 0.074*system + 0.073*survey + 0.071*user + 0.065*time
topic #1 (0.500): 0.105*system + 0.097*graph + 0.096*trees + 0.090*user + 0.088*response
                + 0.087*eps + 0.086*time+ 0.079*survey + 0.071*interface + 0.069*minors
'''

lda.save('/Users/JimberXin/Documents/Github Workingspace/'
         'Natural-Language-Programming/Gensim/Corpus/model.lda')