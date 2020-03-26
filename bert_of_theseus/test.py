# -*- encoding: utf-8 -*-
'''
@Author  :   wangjian
'''
from tensorflow.contrib import predictor
from bert import tokenization
import time

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class SimModel:
    def __init__(self):
        self._load_model()

    def _load_model(self):
        model_dir = './out_2_2_pb/1585107852'
        vocab_file = '/home/wangjian0110/myWork/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
        self.predict_fn = predictor.from_saved_model(model_dir)
        self.tokenizer = tokenization.FullTokenizer(vocab_file)
        self.max_seq_length = 128

    def _convert_single_example(self, text_a, text_b):
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        #while len(input_ids) < self.max_seq_length:
        #    input_ids.append(0)
        #    input_mask.append(0)
        #    segment_ids.append(0)
        return input_ids, input_mask, segment_ids

    def get_feed_dict(self, text_a, text_b_s):
        max_length = 0
        tmp_res = []
        for text_b in text_b_s:
            res = self._convert_single_example(text_a, text_b)
            max_length = max(max_length, len(res[0]))
            tmp_res.append(res)
        feed_dict = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
        for res in tmp_res:
            input_ids, input_mask, segment_ids = res
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            feed_dict['input_ids'].append(input_ids)
            feed_dict['input_mask'].append(input_mask)
            feed_dict['segment_ids'].append(segment_ids)
        return feed_dict

    def similarity(self, text_a, text_b_s):
        if isinstance(text_b_s, str):
            text_b_s = [text_b_s]
        feed_dict = self.get_feed_dict(text_a, text_b_s)
        res = self.predict_fn(feed_dict)
        return res


if __name__ == "__main__":
    model = SimModel()
    res = model.similarity('好', '不好1')
    print(res)
    t1=time.time()
    res = model.similarity('好'*60, ['不'*60]*10)
    print(time.time()-t1)
    t1=time.time()
    res = model.similarity('好'*60, ['不'*60]*10)
    print(time.time()-t1)
    
    
