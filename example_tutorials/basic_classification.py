
# tutorial url: https://www.tensorflow.org/tutorials/keras/basic_classification

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# obtem os dados para o treinamento e testes
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# configura camadas da rede
# camada 1: formata os dados de entrada
# camada 2: cria 128 neuronios para o aprendizado
# camada 3: retorna o resultado com um array de 10 possibilidades
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

# configura o compilador do modelo
model.compile(optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics = ['accuracy'])

# executa o aprendizado
model.fit(train_images, train_labels, epochs = 5)

# testa modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)

print(predictions[0])

maxConfidence = np.argmax(predictions[0])

print('Type: ', class_names[maxConfidence])