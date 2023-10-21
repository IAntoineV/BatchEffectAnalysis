import tensorflow as tf
import numpy as np
from parser_CIFAR import *
from random import shuffle
from batch_management import *
from representation import graph

params = {'CIFAR' : 10}

def residual(x,num_filter):
    skip= x
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([skip, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def create_Resnet(num_residuals, num_filer):

    input = tf.keras.Input(shape=(32, 32, 3))

    x = input
    x = tf.keras.layers.Conv2D(num_filer, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(num_residuals):
        x = residual(x, num_filer)
    x = tf.keras.layers.Conv2D(num_filer, (3, 3), padding='same',activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    output = tf.keras.layers.Dense(units=params['CIFAR'], activation='softmax')(x)
    return tf.keras.Model(inputs=input, outputs=output)

    
class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3))
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=params['CIFAR'], activation='softmax')
        
    def call(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x= self.conv4(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x
    

loss_function = tf.keras.losses.CategoricalCrossentropy()
x_train, x_test, y_train, y_test, names = get_x_y(params['CIFAR'])

def stats(generator_for_fit, nb_classes, nb_test, nb_epoch):
    
    stock=[]
    for i in range(nb_test):
        classes = np.random.choice(params['CIFAR'], size=nb_classes, replace=False)
        print('test', i, sorted(classes))
        resnet = create_Resnet(15,16)
        resnet.build(input_shape=(None, 32, 32, 3))
        resnet.summary()
        optimizer = tf.keras.optimizers.Adam(1e-6, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
        resnet.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = resnet.fit(generator_for_fit(classes), epochs=nb_epoch, validation_data=(x_test, y_test), shuffle = False,)
        stock.append(history.history)
    return stock


def launch_test(nb_classes, nb_test, nb_epoch, list_of_generators):
    list_of_stock = []
    for elt in list_of_generators:
        list_of_stock.append(stats(elt['gen'], nb_classes, nb_test, nb_epoch))
    graph(list_of_stock, [elt['name'] for elt in list_of_generators])
    
if __name__ == '__main__':
    list_of_generators = []
    shuffler = lambda x : CIFARSequence_shuffle(x_train, y_train, x, params['CIFAR'])
    uniformer = lambda x : CIFARSequence_uniform(x_train, y_train, x, params['CIFAR'])
    focuser = lambda x : CIFARSequence_focus(x_train, y_train, x, params['CIFAR'],10, focus_mode)
    full_focuser = lambda x : CIFARSequence_focus(x_train, y_train, x, params['CIFAR'],10, fullfocus_mode)
    list_of_generators.append({
                                'gen' : shuffler,
                                'name': 'shuffle'})
    list_of_generators.append({ 
                               'gen' : uniformer,
                               'name': 'uniform'})
    launch_test(nb_classes = 10, nb_test = 4, nb_epoch = 30, list_of_generators = list_of_generators)