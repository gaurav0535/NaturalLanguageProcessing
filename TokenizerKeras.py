from keras.preprocessing.text import Tokenizer

samples = ["the cat sat on the mat","The dog ate my homework"]


tokenizer = Tokenizer(num_words=100)

tokenizer.fit_on_texts(samples)

print(tokenizer.texts_to_sequences(samples))

print(tokenizer.word_index)

print(tokenizer.texts_to_matrix(samples))
