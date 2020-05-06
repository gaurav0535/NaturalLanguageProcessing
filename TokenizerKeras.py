from keras.preprocessing.text import Tokenizer

samples = ["the cat sat on the mat",
           "The dog ate my homework",
           "My name is gaurav srivastava and i live Noida fslfna"]


tokenizer = Tokenizer(num_words=100,oov_token="Checkit")

tokenizer.fit_on_texts(samples)

#print(tokenizer.texts_to_sequences(samples))

test = ["My name is Gaurav",
        "My name is fdalbfal kvbkk"]

#print(tokenizer.word_index)

#print(tokenizer.texts_to_sequences(test))

################ Padding ##################

from keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(tokenizer.texts_to_sequences(test),padding="post",maxlen=20)
print(padded)


padded = pad_sequences(tokenizer.texts_to_sequences(test),maxlen=20)

print(padded)