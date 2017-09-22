from pprint import pprint
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

'''
String 转换成vector
'''
from gensim import corpora

documents=["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
              "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
             
#去掉common words 并分词
stoplist=set('for a of the and to in'.split()) 
texts=[[word for word in document.lower().split() if word not in stoplist] 
        for document in documents]

#去掉只出现一次的单词
from collections import defaultdict
frequery=defaultdict(int)
#统计词频
for text in texts:
    for token in text:
        frequery[token]+=1
        
texts=[[token for token in text if frequery[token]>1]
       for text in texts]
       
pprint(texts)

#单词转换成id
dictionary=corpora.Dictionary(texts)
dictionary.save('deerwester.dict') #保存dictionary
print(dictionary)
print(dictionary.token2id)

#本文转换成向量, doc2bow简单的统计词频
new_doc="Human computer interaction"
new_vec=dictionary.doc2bow(new_doc.lower().split())
pprint(new_vec)

corpus=[dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('derrwester.mm',corpus) #保存到磁盘
pprint(corpus)

'''
迭代的读取文件中的文本
'''
from six import iteritems
#统计所有的tokens
dictionary=corpora.Dictionary(line.lower().split() for line in open('mycorpora.txt'))
#去掉停用词和只出现一次的词
stop_ids=[dictionary.token2id[stopword] for stopword in stoplist
          if stopword in dictionary.token2id]
once_ids=[tokenid for tokenid,docfreq in iteritems(dictionary.dfs) if docfreq==1]
dictionary.filter_tokens(stop_ids+once_ids) #去掉停用词和只出现一次的词
dictionary.compactify() # 去掉序列之间的空隙
print(dictionary)


'''转换成numpy和scipy矩阵'''
import gensim
import numpy  as np
numpy_matrix=np.random.randint(10,size=[5,2])
#dense2corpus
corpus=gensim.matutils.Dense2Corpus(numpy_matrix)
#corpus2dense
numpy_matrix=gensim.matutils.corpus2dense(corpus,num_terms=2)











