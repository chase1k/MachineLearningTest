import pandas as pd

dataset = pd.read_csv('cancer.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #, random_state=1)

import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(1024, activation='sigmoid')) #Change the number here and 1 below
model.add(tf.keras.layers.Dense(1024, activation='sigmoid')) #Make sure it is a power of two
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

model.fit(x_train, y_train, epochs=100) #Change to alter how many times it runs after changing
model.evaluate(x_test, y_test)
