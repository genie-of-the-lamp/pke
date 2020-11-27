import sys
import re

from pke.readers import Reader
from pke.data_structures import Document

import lucene
from org.apache.lucene.analysis.ko import KoreanAnalyzer, KoreanTokenizer
from org.apache.lucene.analysis.ko.tokenattributes import PartOfSpeechAttribute
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, OffsetAttribute
from java.util import HashSet


class KoreanToken(object):
    def __init__(self, idx, text, lemma, pos, offset):
        self.idx = idx
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.offset = offset


class KoreanDocument(object):
    def __init__(self, text, userdict=None):
        if not lucene.getVMEnv():
            lucene.initVM()
        self.analyzer = KoreanAnalyzer(userdict, KoreanTokenizer.DecompoundMode.DISCARD, HashSet(), False)

        sents = pre_proc_with_split_sent(text)
        self.sents = [self.get_tokens(sent) for sent in sents]

    def get_tokens(self, sent):
        tokens_info = self.analyze(sent)
        tokens = []
        for idx, info in enumerate(tokens_info):
            pos = info['left_POS']
            if pos != info['right_POS']:
                pos = "{}+{}".format(pos, info['right_POS'])
            offset = (info['start_offset'], info['end_offset'])
            token = KoreanToken(idx, info['token'], info['token'], pos, offset)
            tokens.append(token)
        return tokens

    def analyze(self, text):
        """
        params:
            text: (str) the docuemnt to be analyzed.
        return:
            result: (list) the list of token attributes.
            ex)
            [
                {
                    'token': '가곡역', 
                    'start_offset': 0, 
                    'end_offset': 3, 
                    'left_POS': 'NNP', 
                    'right_POS': 'NNP', 
                }, 

                 ...
            ] 
        """
        ts = self.analyzer.tokenStream("", text)
        term_attr = ts.addAttribute(CharTermAttribute.class_)
        offset_attr = ts.addAttribute(OffsetAttribute.class_)
        morph_attr = ts.addAttribute(PartOfSpeechAttribute.class_)
        ts.reset()

        result = []

        while ts.incrementToken():
            result.append({  
                'token': term_attr.toString(),  
                'start_offset': offset_attr.startOffset(), 
                'end_offset': offset_attr.endOffset(), 
                'left_POS': morph_attr.getLeftPOS() and morph_attr.getLeftPOS().name() or None,
                'right_POS': morph_attr.getRightPOS() and morph_attr.getRightPOS().name() or None,
                })

        ts.end()
        ts.close()

        return result


class KoreanRawTextReader(Reader):
    def __init__(self, **kwargs):
        self.userdict = kwargs.get('userdict', None)

    def read(self, text, **kwargs):
        """
        params:
            text: (str) raw text to pre-process.
        """
        kr_model_doc = KoreanDocument(text, self.userdict)
        sentences = []
        for sentence_id, sentence in enumerate(kr_model_doc.sents):
            sentences.append({
                'words': [token.text for token in sentence],
                'lemmas': [token.lemma_ for token in sentence], 
                'POS': [token.pos_ for token in sentence],
                'char_offsets': [token.offset for token in sentence]
            })

        doc = Document.from_sentences(sentences, 
                                      input_file=kwargs.get('input_file', None),
                                      **kwargs)
        return doc

seperator = '．.!?'

def pre_proc_kor(text):
    result = []
    if type(text) == str: 
        text = [text]
    
    for t in text:
        t = re.sub("[^가-힣a-z1-9{}]".format(seperator)," ", t)
        t = re.sub("[ ]+", " ", t)
        t = t.strip()
        if t: result.append(t)

    return result

def pre_proc_with_split_sent(text, userdict=None):
    text = " {} ".format(text.lower())
    text = re.sub(r'\s가-힣{1}.]+가-힣{1}(\s|[{}])'.format(seperator),
        lambda x: x.group()[:-1].replace('.', '') + x.group[-1], text)
    text = re.sub(r'[{}]+'.format(seperator), lambda x: x.group()[0]+"\n", text)
    text = re.split('\n', text)
    text = pre_proc_kor(text)

    return text

