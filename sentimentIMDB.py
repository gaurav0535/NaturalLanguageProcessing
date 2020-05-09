import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

print(tf.__version__)

imdb,info = tfds.load("imdb_reviews",with_info=True,as_supervised=True)

train_data ,test_data = imdb["train"],imdb["test"]
print(train_data)
print(test_data)