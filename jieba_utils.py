import jieba
import jieba.analyse

puncts = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

stop_wordset = puncts

print("Preparing jieba utils...")

jieba.set_dictionary('dict.txt.big')

with open('stop_words.txt', 'r', encoding='utf-8') as f:
    for stop_word in f:
        stop_wordset.add(stop_word.strip('\n'))

print("Jieba utils prepared.")


def cut(sentence, cut_all=False):
    return [word for word in jieba.cut(sentence, cut_all=cut_all)
            if len(word) != 0
            and word not in stop_wordset]


def extract_tag(sentence, withWeight=False):
    return jieba.analyse.extract_tags(sentence, withWeight=withWeight)

