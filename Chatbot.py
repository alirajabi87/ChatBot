"""
Simple Chat Bot as personal project.
This simple Chatbot uses a single layer of "hops", in the next version,
I will add more hops layers to the chatbot main model.

I also added my trained model and answer test, To use them, please read the comments
in the App. I hope you enjoy it.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To prevent Tensorflow Warnings (I don't like them)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, add, dot, Dense, Dropout, Permute, Embedding, concatenate, Activation, LSTM

with open('train_qa.txt', 'rb') as file:
    train_data = pickle.load(file)

with open('test_qa.txt', 'rb') as file:
    test_data = pickle.load(file)

# # Test the data:
# print(train_data[0])

all_data = test_data + train_data

vocab = set()
for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

# Add answer words
# (The positions of the 'yes' and 'no' are always random,
# so if you want to check the accuracy agter loading
# the model, please uncomment the appropriate sections

vocab.add('no')
vocab.add('yes')
vocab_len = len(vocab) + 1  # one is a place holder for keras pad_sequences

# Longest Story and question
max_story_lens = max([len(data[0]) for data in all_data])
max_question_lens = max([len(q[1]) for q in all_data])

tokenizer = Tokenizer(filters=[]) # For this dataset, we don't use filters an empty string or list is OK.
tokenizer.fit_on_texts(vocab)

# print(tokenizer.word_index)

train_story_text = []
train_question_text = []
train_answer = []

for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answer.append(answer)

# # If you need these, you can uncomment them, I changed the code a bit, so, I haven't used them
# train_story_seq = tokenizer.texts_to_sequences(train_story_text)
# train_question_seq = tokenizer.texts_to_sequences(train_question_text)
# train_answer_seq = tokenizer.texts_to_sequences(train_answer)


def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_lens,
                      max_question_len=max_question_lens):
    X = []
    Xq = []
    Y = []
    # print(word_index)
    for story, question, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


input_train, queries_train, answer_train = vectorize_stories(train_data)
input_test, queries_test, answer_test = vectorize_stories(test_data)


# Creating the Neural Network (Based on paper "End-To-End Memory Networks")
# input encoder A (Story)
# input encoder C (Story)
# input encoder Q (Query)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Create place holders --> shape = (max_story_length, batch_size), the batch_size will be determined by the model itself

input_sequence = Input(shape=(max_story_lens,))
input_question = Input(shape=(max_question_lens,))

# Input Encoder A

input_encoder_A = Sequential()
input_encoder_A.add(Embedding(input_dim=vocab_len, output_dim=64))  # output from the paper
input_encoder_A.add(Dropout(0.4))
# Output --> (samples, story_maxlen, embeding_dim)

# Input Encoder C

input_encoder_C = Sequential()
input_encoder_C.add(Embedding(input_dim=vocab_len, output_dim=max_question_lens))
input_encoder_C.add(Dropout(0.4))

# Output --> (samples, story_maxlen, max_question_len)

# Input Encoder Q

input_encoder_Q = Sequential()
input_encoder_Q.add(Embedding(input_dim=vocab_len, output_dim=64,
                              input_length=max_question_lens))  # output matchs to the Encoder_A and input_length matches to Question_length
input_encoder_Q.add(Dropout(0.4))

# Output --> (samples, story_maxlen, embeding_len)

# -------------------------------------------------------------------------------------------- #
# Encoded <----- ENCODER(INPUT)
input_encoded_A = input_encoder_A(input_sequence)
input_encoded_C = input_encoder_C(input_sequence)
input_encoded_Q = input_encoder_Q(input_question)

# --------------------------------------------------------------------------------------------- #

match = dot([input_encoded_A, input_encoded_Q], axes=(2, 2))
match = Activation('softmax')(match)

response = add([match, input_encoded_C])
response = Permute((2, 1))(response)

answer = concatenate([response, input_encoded_Q])
# For Checking: print(answer)

answer = LSTM(32)(answer)
answer = Dropout(0.4)(answer)
answer = Dense(vocab_len)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, input_question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# # To train the model, uncomment the following. To use the Chat Bot model, comment it again
# results = model.fit([input_train, queries_train],
#                     answer_train,
#                     batch_size=16,
#                     epochs=50,
#                     validation_data=([input_test, queries_test], answer_test),
#                     verbose=2)
#
# # Saving the model and the test
# model.save('chatbot.h5')
# pickle.dump(tokenizer, open('tokenizer', 'wb'))
#
# # Uncomment if you want to check the accuracy after loading the model
# pickle.dump(input_test, open('input_test', 'wb'))
# pickle.dump(queries_test, open('queries_test', 'wb'))
# pickle.dump(answer_test, open('answer_test', 'wb'))
#
#
# plt.plot(results.history['accuracy'], label='accuracy')
# plt.plot(results.history['val_accuracy'], label='val_accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.title('Accuracy During the Training')
# plt.legend()
# plt.show()

model = load_model('chatbot.h5')
tokenizer = pickle.load(open('tokenizer', 'rb'))

# # Uncomment if you want to check the accuracy after loading the model
# answer_test = pickle.load(open('answer_test', 'rb'))
# input_test = pickle.load(open('input_test', 'rb'))
# queries_test = pickle.load(open('queries_test', 'rb'))

# # For checking the accuracy, Please uncomment the following before training
# predictions = model.predict(([input_test, queries_test]))
# predictions = np.argmax(predictions, axis=-1)
# answer_test = np.argmax(answer_test, axis=-1)
# df = pd.DataFrame(confusion_matrix(answer_test, predictions), index=['No', 'Yes'], columns=['No', 'yes'])
# print(df)
# print(classification_report(answer_test, predictions))

myStory = "John left the kitchen . Sandra dropped the football in the garden ."
myQuestion = "Is the football in the garden ?"

myData = [(myStory.split(), myQuestion.split(), 'yes')]
myStory, my_Ques, my_ans = vectorize_stories(myData)

singlePred = np.argmax(model.predict([myStory, my_Ques]))
print(myQuestion)
print(tokenizer.index_word[singlePred])

