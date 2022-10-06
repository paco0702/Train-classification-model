import flask
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

reloaded_model = tf.keras.models.load_model(
        "official_comment_model.h5")



vocab_size = 7000
oov_tok = "<OOV>"
trunc_type = 'post'
pad_type = 'post'
max_length = 87
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)


app = flask.Flask(__name__)
comment = ''


@app.route("/perform_comment_analysis", methods=['GET','POST'])
def getComment():
    comment = flask.request.headers['comment']
    tokenizer = getTokenizer()
    print(comment)
    reviews = [comment]
    padding_type = 'post'
    sample_sequences = tokenizer.texts_to_sequences(reviews)
    reviews_padded = pad_sequences(sample_sequences, padding=padding_type,
                                   maxlen=max_length)
    print(reviews_padded)
    classes = reloaded_model.predict(reviews_padded)
    float_array = classes.astype(np.float)
    result_array = float_array[0].tolist()
    result = "{:.2f}".format(result_array[0])
    print(result)
    return result


def getTokenizer():
    training_reviews = []
    df = pd.read_csv('C:/Users/Pacowawo Chiu/PycharmProjects/helloWorld/FYP/training_review.csv')
    training_len = 5867
    for i in range(training_len):
        item = df['training_reviews'][i]
        training_reviews.append(item)

    tokenizer.fit_on_texts(training_reviews)
    word_index = tokenizer.word_index
    return tokenizer


if __name__ == "__main__":
    app.run(host="0.0.0.0")
