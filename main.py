from transformers import BertTokenizer, AutoTokenizer

from bert_model import BertSeqCls
from tensorflow import keras
import tensorflow as tf
import numpy as np

# [[-1.6710131,  3.3942356, -1.5105639]]

if __name__ == '__main__':
    def test():
        model = BertSeqCls()
        model = model.load()
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        inputs = tokenizer('這部電影很好看', '推！真的很讚',
                           add_special_tokens=True,
                           max_length=512)
        input_ids, input_segments, input_masks = [], [], []
        input_ids.append(inputs['input_ids'])
        input_segments.append(inputs['token_type_ids'])
        input_masks.append(inputs['attention_mask'])

        input_ = [np.array(input_ids, dtype=np.int32),
                  np.array(input_segments, dtype=np.int32),
                  np.array(input_masks, dtype=np.int32)]
        print(input_)
        pre = model.predict(input_)
        print(pre)

    model = BertSeqCls()
    pre = model.predict(['這部電影很好看'], ['推！真的很讚'])
    print(pre)
