import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sentencepiece as spm
import pandas as pd
import os
import gzip
import shutil

app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

def data_preprocessing(train, test):
    vocab_file = "./kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    x_train = train["comments"]
    y_train = train["label"]
    x_test = test["comments"]
    y_test = test["label"]

    for l in range(len(x_train)):
        x_train[l] = vocab.encode_as_ids(x_train[l])

    for l in range(len(x_test)):
        x_test[l] = vocab.encode_as_ids(x_test[l])

    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    return x_train, y_train, x_test, y_test

def evalution(x_test, model):

    # data_pre = model.predict(x_test)
    # print(data_pre)
    pred_roc = model.predict(x_test)
    pred_z = []
    for j in range(len(pred_roc)):
        pred_z.append(list(pred_roc[j]).index(max(list(pred_roc[j]))))

    pred_z = np.asarray(pred_z).astype('float32')

    return pred_z

# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기
        x_test = request.form['input_text']
        if not x_test:
            return render_template('index.html', label="No Text")

        line_input = []
        line_input.append(str(x_test))
        lines = pd.DataFrame({"comments": line_input, "label": [0]})
        line_test = lines.copy()

        _, _, line_x, _ = data_preprocessing(line_test, lines)

        line_x = keras.preprocessing.sequence.pad_sequences(line_x, maxlen=maxlen)

        # 입력 받은 텍스트 예측
        label = evalution(line_x, model)

        # 숫자가 10일 경우 0으로 처리
        if label[0] == 0: out_label = 'Good'
        elif label[0] == 1: out_label = 'Bias'
        print(out_label)

        # 결과 리턴
        return render_template('index.html', label=out_label)


def make_model(maxlen, vocab_size):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성

    model = make_model(maxlen, vocab_size)
    model.load_weights('./model/checkpoints_model_trans3.ckpt')
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
