import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from dataHelper import read_dataset


class BertSeqCls:
    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        self.batch_size = 5
        self.checkpoint_path = 'bertSeqCls/movie_comment.hdf5'

    def map_example_to_dict(self, input_ids, attention_masks, token_type_ids, label):
        return {
                   "input_ids": input_ids,
                   "token_type_ids": token_type_ids,
                   "attention_mask": attention_masks,
               }, label

    def fit(self, main_contents, comments, labels):
        inputs = self.tokenizer(main_contents, comments, return_tensors="tf", max_length=512,
                                padding='max_length', truncation=True)
        labels = tf.reshape(np.array(labels), (-1, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], labels))
        ds_train_encoded = train_dataset.map(self.map_example_to_dict).batch(self.batch_size)

        learning_rate = 2e-5
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        number_of_epochs = 10
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        early_stop_callback = EarlyStopping(
            monitor='accuracy',
            min_delta=0.0001,  # 精確度至少提高0.0001
            patience=3)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            save_weights_only=True,
            monitor='accuracy',
            mode='max',
            save_best_only=True,
            verbose=1)

        self.model.fit(ds_train_encoded, epochs=number_of_epochs,
                       callbacks=[early_stop_callback, model_checkpoint_callback])

    def train(self, data_path):
        main_contents, comments, labels = read_dataset(data_path)
        self.fit(main_contents, comments, labels)

    def load(self):
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        self.model.load_weights(self.checkpoint_path)

    def map_test_example_to_dict(self, input_ids, attention_masks, token_type_ids):
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks,
        }

    def predict(self, main_contents, comments):
        inputs = self.tokenizer(main_contents, comments, return_tensors="tf", max_length=512,
                                padding='max_length', truncation=True)
        bert_input = tf.data.Dataset.from_tensor_slices(
            (inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])).map(self.map_test_example_to_dict)

        bert_input = bert_input.batch(self.batch_size)
        self.load()
        predicts = self.model.predict(bert_input)[0]
        pred = []
        for pre in predicts:
            pred.append(np.argmax(pre))
        return pred
