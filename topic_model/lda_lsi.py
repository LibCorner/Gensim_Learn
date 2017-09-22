import jieba
import re
from gensim import models
from six import iteritems
#coding:utf-8
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint

stop_words=['是','的']


def topicModel(content,topic_num=10):
    '''主题模型
    '''
    texts=content#re.split("。",content)
    #texts=[c[1] for c in content]
    #print(texts)
    vec=CountVectorizer(ngram_range=(1,3),min_df=1,tokenizer=jieba.lcut) #n-gram
    analyzer=vec.build_analyzer()
    #texts=[jieba.lcut(text) for text in texts]
    texts=[[t.replace(" ","") for t in analyzer(text)] for text in texts]
           
    dictionary=corpora.Dictionary(texts)
    #去掉只出现一次的词 和停用词
    once_ids=[w for w,df in iteritems(dictionary.dfs) if df==1 and df>len(texts)*0.7]
    stop_ids=[dictionary.token2id[w] for w in stop_words if w in dictionary.token2id]
    dictionary.filter_tokens(stop_ids+once_ids)
    dictionary.compactify()
    
    corpus=[dictionary.doc2bow(text) for text in texts]
    
    #tfIdf model
    tfidf=models.TfidfModel(corpus)
    tfidf_corpus=tfidf[corpus]
    
    #lsi model
    lsi_model=models.LsiModel(tfidf_corpus,num_topics=topic_num,id2word=dictionary)
    topics_lsi=lsi_model.print_topics(5)
    pprint(topics_lsi)
    
    lda=models.LdaModel(corpus,num_topics=topic_num,id2word=dictionary)
    topics_lda=lda.print_topics(5)
    pprint(topics_lda)
    
if __name__=="__main__":
    content=["你 最近 光顾 着 玩 了 ， 没有 注意 到 我 微博 里 的 新 动向 。",
            "不是 阿 …",
            "这 诱惑 不得了 了 ！ ！",
            "就是 不 信任 我",
            "有 一个 字 叫 排队 ， 下次 你 上班 ， 也许 也 可以 这样 要求 别人 ， 特别 是 医生",
            "曹 老师 ， 可否 介绍 几 个 知识 产权 界 名 律师 ？",
            "她 跟 小亮 真的 好 衬",
            "销售 与 市场 礼品 版 总编 ， 来看 你 的 了 哦 ， 咱们 互相 关注 哈 ~",
            "淡定 的 告诉 你 ， 我 已经 穿 薄 羊毛 裤 了",
            "貌似 很 不错 . . .",
            "今晚 我 这里 阴天 ！ ！",
            "那 鼻子 真是 亮点",
            "其实 这种 天气 是 很 适合 睡觉 的 ， 我 这 人 讨厌 死 了 ， 在家 不 睡 ， 现在 却 很 困",
            "谁 的 啊 ， 这 大 魅力 啊 ？"]
    topicModel(content,topic_num=10)