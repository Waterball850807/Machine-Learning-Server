import logging

import jieba
import numpy
from gensim.corpora import WikiCorpus
from gensim.models import word2vec


def load_all_wiki_articles_into_txt():
    wiki_corpus = WikiCorpus('zhwiki-20180120-pages-articles.xml.bz2', dictionary={})
    texts_num = 0

    with open("wiki_texts.txt", 'w', encoding='utf-8') as output:
        for text in wiki_corpus.get_texts():
            output.write(' '.join(text) + '\n')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已處理 %d 篇文章" % texts_num)


def cut_words_from_wiki_zh_tw():
    # jieba custom setting.
    jieba.set_dictionary('dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('stop_words.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open('wiki_seg.txt', 'w+', encoding='utf-8')
    with open('wiki_zh_tw.txt', 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()


def train_word2vec_model():
    sentences = word2vec.LineSentence("wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=get_word_vector_size())
    model.save("word2vec.model")


my_model = None


def get_word2vec_model():
    global my_model
    if not my_model:
        my_model = word2vec.Word2Vec.load("D:\開發套件專區\機器學習\pretrained word2vec\zh.bin")
    return my_model


def get_word_vector_size():
    return 300


def get_fast_text_word2vec_model():
    from gensim.models.wrappers import FastText

    global my_model
    if not my_model:
        print("Loading word2vec model...")
        my_model = FastText.load_fasttext_format("D:\開發套件專區\機器學習\wiki.zh")
        print("Loading word2vec model loaded.")
    return my_model


def get_fast_text_word_vector_size():
    return 300


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = get_word2vec_model()
    while True:
        try:
            word1, word2 = input('input two words: ').split()
            print(model.wv.similarity(word1, word2))
        except:
            print("Error!")


if __name__ == '__main__':
    main()
