

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow.",
    "Professor Plum has a green plant in his study.",
    "Miss Scarlett watered Professor Plum's green plant while he was away from his office last week."
]


print(len(sentences))

tokenizer = Tokenizer( num_words = 51)

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index 
word_index
print(len(word_index))


# sparse matrix
count = tokenizer.texts_to_matrix(sentences, mode='count')
count.shape
print('count', count)

tfidf = tokenizer.texts_to_matrix(sentences, mode='tfidf')
print('tfidf', tfidf) 



seq_vector = tokenizer.texts_to_sequences(sentences) 
print(seq_vector)
'''
[[2, 1, 10, 11, 12, 3, 4, 5, 13, 4, 14, 2, 1, 15, 16, 6, 17, 18, 19],
 [7, 20, 21, 6, 1, 8, 3, 9, 5],
 [22, 23, 24, 7, 25, 1, 8, 26, 27, 28, 29, 30, 9, 31, 32, 33]]
'''

lens = [len(tx) for tx in seq_vector]
maxlen = 16


x_data = pad_sequences(seq_vector, maxlen=maxlen)
print(x_data.shape)
print(x_data)














