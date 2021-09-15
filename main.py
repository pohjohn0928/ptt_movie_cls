import requests
from transformers import BertTokenizer, AutoTokenizer

from bert_model import BertSeqCls
from tensorflow import keras
import tensorflow as tf
import numpy as np

# [[-1.7695713  2.4140973 -1.187638 ]]

if __name__ == '__main__':
    def test():
        model = BertSeqCls()
        # bert_model = model.load()
        # bert_model.save_pretrained('bert_model', saved_model=True)
        pre = model.predict(['這部電影很好看'], ['真的很讚'])
        print(pre)
        # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        # inputs = tokenizer('這部電影很好看', '推！真的很讚',
        #                    add_special_tokens=True,
        #                    max_length=512)
        # model.from_pretrained()

    def tf_serving_test():
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        inputs = tokenizer('這部電影很好看', '推！真的很讚', max_length=512,
                           padding='max_length', truncation=True)

        input_dic = {'input_ids': inputs['input_ids'],
                     'token_type_ids': inputs['token_type_ids'],
                     'attention_mask': inputs['attention_mask']}

        batch = [dict(input_dic)]
        input_data = {'instances': batch}
        print(input_data)
        r = requests.post("http://localhost:8501/v1/models/bert_model:predict", data=json.dumps(input_data))
        print(r.json())
