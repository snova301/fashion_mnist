# -*- coding: utf-8 -*- #

import datetime
dt_start = datetime.datetime.now()
print('start time : ' + dt_start.strftime('%Y-%m-%d %H:%M:%S'))

import time
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

import optuna
from keras.datasets import fashion_mnist
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger
from keras.utils import to_categorical, plot_model


# --- Global variable --- #
Batch_size = 128
Epochs = 20


class MNIST_CNN:
    def pre_proc(self, list_n_data):
        # --- load fashion_mnist data --- #
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # --- reducing data --- #
        x_train, y_train, x_test, y_test = self.reduce_data(x_train, y_train, x_test, y_test, list_n_data)

        # --- sample image --- #
        # self.img_save(x_train, y_train)

        # --- covert train input --- #
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

        # --- convert one-hot vector --- #
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # --- save fashion_mnist data --- #
        np.save('x_train.npy', x_train)
        np.save('x_test.npy', x_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

        return x_train, y_train, x_test, y_test


    def reduce_data(self, x_train, y_train, x_test, y_test, list_n_data):
        x_train = x_train[list_n_data[0]:list_n_data[1]]
        y_train = y_train[list_n_data[0]:list_n_data[1]]
        x_test = x_test[list_n_data[2]:list_n_data[3]]
        y_test = y_test[list_n_data[2]:list_n_data[3]]

        return x_train, y_train, x_test, y_test


    def img_save(self, x_dat, y_dat):
        for i_img, x_img in enumerate(x_dat):
            img_fashion_mnist = Image.fromarray(np.uint8(x_img))
            img_fashion_mnist.save(str(y_dat[i_img])+'_'+str(i_img)+'.png')
            if i_img > 10:
                break


    def plot_result(self, history):
        # accuracy
        plt.figure()
        plt.plot(history.history['acc'], label='acc', marker='.')
        plt.plot(history.history['val_acc'], label='val_acc', marker='.')
        # plt.grid()
        plt.legend(loc='best')
        plt.title('accuracy')
        plt.savefig('graph_accuracy.png')
        # plt.show()

        # loss
        plt.figure()
        plt.plot(history.history['loss'], label='loss', marker='.')
        plt.plot(history.history['val_loss'], label='val_loss', marker='.')
        # plt.grid()
        plt.legend(loc='best')
        plt.title('loss')
        plt.savefig('graph_loss.png')
        # plt.show()


    def keras_model_saving(self, model, model_filename):
        model.save(model_filename)


    def keras_model_loading(self, model_filename):
        return load_model(model_filename, compile=False)


    def create_model(self, x_train, y_train, x_test, y_test):
        # --- create model --- #
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # --- start learning --- #
        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

        print(model.summary())

        # --- callback function --- #
        csv_logger = CSVLogger('trainlog.csv')

        # --- train --- #
        history = model.fit(x_train, y_train,
                            batch_size=Batch_size, epochs=Epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[csv_logger])

        # --- result --- #
        self.evaluate_model(model, x_train, y_train, x_test, y_test)
        self.predict_model(model, x_train, y_train, x_test, y_test)
        self.keras_model_saving(model, 'mnist_model.hdf5')
        # self.plot_result(history)
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB') # TB or LR


    def evaluate_model(self, model, x_train, y_train, x_test, y_test):
        # --- result --- #
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss: {0}'.format(score[0]))
        print('Test accuracy: {0}'.format(score[1]))


    def predict_model(self, model, x_train, y_train, x_test, y_test):
        pred_train = model.predict(x_train, verbose=0)
        pred_test = model.predict(x_test, verbose=0)

        # --- for train --- #
        arg_pred_train = np.argmax(pred_train, axis=1)
        arg_y_train = np.argmax(y_train, axis=1)
        acc_train = np.sum(arg_pred_train - arg_y_train == 0)/pred_train.shape[0]
        print('train accuracy : ' + str(acc_train))

        # --- for test --- #
        arg_pred_test = np.argmax(pred_test, axis=1)
        arg_y_test = np.argmax(y_test, axis=1)
        acc_test = np.sum(arg_pred_test - arg_y_test == 0)/pred_test.shape[0]
        print('test accuracy : ' + str(acc_test))


    def tranfer_learning(self, x_train, y_train, x_test, y_test):
        trans_model = self.keras_model_loading('mnist_model.hdf5')
        self.predict_model(trans_model, x_train, y_train, x_test, y_test)


    def fine_tune_model(self, x_train, y_train, x_test, y_test):
        tune_model = self.keras_model_loading('mnist_model.hdf5')
        tune_model.pop() # delete softmax

        for loop_layer, model_layer in enumerate(tune_model.layers):
            print('layer : ' + str(loop_layer))
            model_layer.trainable = False
            print(model_layer.get_config())

        print(tune_model.summary())
        print(tune_model.output)

        # --- add layer --- #
        top_model = Sequential()
        top_model.add(Dense(128, activation='relu', name='top_dense1'))
        top_model.add(Dropout(0.5, name='top_dropout'))
        top_model.add(Dense(10, activation='softmax', name='top_dense2'))

        tune_model = Model(inputs=tune_model.input, outputs=top_model(tune_model.output))

        tune_model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

        print(tune_model.summary())

        # --- callback function --- #
        csv_logger = CSVLogger('trainlog_tune.csv')

        # --- train --- #
        history = tune_model.fit(x_train, y_train,
                            batch_size=Batch_size, epochs=Epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[csv_logger])

        # -- result --- #
        self.predict_model(tune_model, x_train, y_train, x_test, y_test)
        self.keras_model_saving(tune_model, 'tune_model.hdf5')
        plot_model(tune_model, to_file='tune_model.png', show_shapes=True, show_layer_names=True, rankdir='LR') # TB or LR


    def main(self):
        # --- preparation --- #
        list_n_data = [0, 1000, 0, 1000] # [start_train, end_train, start_test, end_test]
        x_train, y_train, x_test, y_test = self.pre_proc(list_n_data)

        # --- create model --- #
        # self.create_model(x_train, y_train, x_test, y_test)

        # --- fine tuning --- #
        list_n_data = [1000, 2100, 1000, 2100] # [start_train, end_train, start_test, end_test]
        x_train, y_train, x_test, y_test = self.pre_proc(list_n_data)
        # self.tranfer_learning(x_train, y_train, x_test, y_test)
        self.fine_tune_model(x_train, y_train, x_test, y_test)

        # load_tune_model = load_model('tune_model.hdf5', compile=False)
        # print(load_tune_model.summary())



if __name__ == '__main__':
    MNIST_CNN().main()

    # --- Record Time --- #
    dt_end = datetime.datetime.now()
    dt_delta = dt_end - dt_start

    print('end time : ' + dt_end.strftime('%Y-%m-%d %H:%M:%S'))
    print('diff time : ' + str(dt_delta))

