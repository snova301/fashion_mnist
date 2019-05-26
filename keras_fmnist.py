import time
import numpy as np
import matplotlib.pylab as plt

from keras.datasets import fashion_mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger
from keras.utils import to_categorical, plot_model

# バッチ数とエポック数の定義
Batch_size = 128
Epochs = 20


class FASHION_MNIST:
    def pre_proc(self, s_name):
        # fashion_mnistのロード
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # データを抽出
        if s_name == '1st':
            list_n_data = [0, 1000, 0, 1000] # 0番目から1000番目のデータを抽出
        elif s_name == '2nd':
            list_n_data = [2000, 3000, 2000, 3000] # 2000番目から3000番目のデータを抽出
        x_train = x_train[list_n_data[0]:list_n_data[1]]
        y_train = y_train[list_n_data[0]:list_n_data[1]]
        x_test = x_test[list_n_data[2]:list_n_data[3]]
        y_test = y_test[list_n_data[2]:list_n_data[3]]

        # データ処理
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255
        y_train = to_categorical(y_train, 10) # one-hot
        y_test = to_categorical(y_test, 10)# one-hot

        return x_train, y_train, x_test, y_test


    def plot_result(self, history, s_name):
        # 精度のグラフ
        plt.figure()
        plt.plot(history.history['acc'], label='acc', marker='.')
        plt.plot(history.history['val_acc'], label='val_acc', marker='.')
        plt.ylim(0, 1)
        plt.legend(loc='best')
        plt.title('accuracy')
        plt.savefig('graph_accuracy_'+s_name+'.png')

        # 損失のグラフ
        plt.figure()
        plt.plot(history.history['loss'], label='loss', marker='.')
        plt.plot(history.history['val_loss'], label='val_loss', marker='.')
        plt.legend(loc='best')
        plt.title('loss')
        plt.savefig('graph_loss_'+s_name+'.png')


    def create_model(self):
        # モデルを作る
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        return model


    def create_finetuning_model(self):
        # モデルをロード
        model = load_model('model_1st.hdf5', compile=False)
        model.pop() # delete softmax

        # freeze機能は今回使用しない
        # for model_layer in model.layers:
        #     model_layer.trainable = False

        # 層の追加
        model_a = model.output
        model_a = Dense(128, activation='relu', name='model_a_dense1')(model_a)
        model_a = Dense(10, activation='softmax', name='model_a_dense2')(model_a)

        tuning_model = Model(inputs=model.input, outputs=model_a)

        return tuning_model


    def start_learning(self, model, x_train, y_train, x_test, y_test, s_name):
        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

        csv_logger = CSVLogger('trainlog_'+s_name+'.csv')
        history = model.fit(x_train, y_train,
                            batch_size=Batch_size, epochs=Epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[csv_logger])

        # 結果
        print(model.summary())
        self.plot_result(history, s_name)
        self.predict_model(model, x_train, y_train, x_test, y_test)
        model.save('model_'+s_name+'.hdf5')
        plot_model(model, to_file='model_'+s_name+'.png', show_shapes=True, show_layer_names=True, rankdir='TB')


    def predict_model(self, model, x_train, y_train, x_test, y_test):
        pred_train = model.predict(x_train, verbose=0)
        pred_test = model.predict(x_test, verbose=0)

        # 訓練データ用
        arg_pred_train = np.argmax(pred_train, axis=1)
        arg_y_train = np.argmax(y_train, axis=1)
        acc_train = np.sum(arg_pred_train - arg_y_train == 0)/pred_train.shape[0]
        print('train accuracy : ' + str(acc_train))

        # テストデータ用
        arg_pred_test = np.argmax(pred_test, axis=1)
        arg_y_test = np.argmax(y_test, axis=1)
        acc_test = np.sum(arg_pred_test - arg_y_test == 0)/pred_test.shape[0]
        print('test accuracy : ' + str(acc_test))


    def main(self):
        # 1st learning
        x_train, y_train, x_test, y_test = self.pre_proc('1st')
        model = self.create_model()
        self.start_learning(model, x_train, y_train, x_test, y_test, '1st')

        # fine-tuning
        x_train, y_train, x_test, y_test = self.pre_proc('2nd')
        model = self.create_finetuning_model()
        self.start_learning(model, x_train, y_train, x_test, y_test, '2nd')


if __name__ == '__main__':
    FASHION_MNIST().main()