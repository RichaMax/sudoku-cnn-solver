import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Reshape, Activation, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import datetime


def sudoku_acc(ytrue, ypred):
    ytrue = tf.cast(ytrue, dtype=tf.int64)
    ypred_a = tf.math.argmax(ypred, axis=-1)
    ypred_b = tf.expand_dims(ypred_a, axis=-1)
    r = tf.equal(ytrue, ypred_b)
    r = tf.cast(r, dtype=tf.float32)
    return tf.reduce_mean(r)


def correct_grid(ytrue, ypred):
    ytrue = tf.cast(ytrue, dtype=tf.int64)
    ypred_a = tf.math.argmax(ypred, axis=-1)
    ypred_b = tf.expand_dims(ypred_a, axis=-1)
    r = tf.equal(ytrue, ypred_b)
    r = tf.cast(r, dtype=tf.float32)
    r_s = tf.reduce_sum(r, axis = [1,2])
    return tf.reduce_mean(r_s//81)


def sudoku_loss(ytrue, ypred, from_logits=False):
    l1 = tf.keras.losses.SparseCategoricalCrossentropy()(ytrue, ypred)

    ypred_a = tf.math.argmax(ypred, axis=-1)
    t_cols = tf.equal(tf.reduce_sum(ypred_a, axis=1), 45)
    t_rows = tf.equal(tf.reduce_sum(ypred_a, axis=2), 45)

    l2 = 1.0 - tf.reduce_mean(tf.cast(t_cols, dtype=tf.float32))
    l3 = 1.0 - tf.reduce_mean(tf.cast(t_rows, dtype=tf.float32))
    return l1+l2+l3


def data_generator(x,y):
    for i in range(x.shape[0]):
        yield x[i], y[i]


def make_batches(ds, buffersize, batchsize):
    return ds.cache().shuffle(buffersize).batch(batchsize).prefetch(tf.data.AUTOTUNE)


class CnnLayer(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, dropout_r=0.1, activation='relu', padding='same'):
        super(CnnLayer, self).__init__()
        self.conv = Conv2D(filter_size, kernel_size=(kernel_size,kernel_size), activation=activation, padding=padding)
        self.batchnorm = BatchNormalization()
        self.dropout = Dropout(dropout_r)

    def call(self, inputs, training, **kwargs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        return self.dropout(x, training=training)


class CnnLayer2(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, dropout_r=0.1, activation=tf.nn.relu, padding='same'):
        super(CnnLayer2, self).__init__()
        self.act_func = activation
        self.conv = Conv2D(filter_size, kernel_size=(kernel_size,kernel_size), padding=padding)
        self.batchnorm = BatchNormalization()
        self.dropout = Dropout(dropout_r)
        self.conv2 = Conv2D(filter_size, kernel_size=(kernel_size, kernel_size), padding=padding)
        self.batchnorm2 = BatchNormalization()
        # self.dropout2 = Dropout(dropout_r)
        self.conv3 = Conv2D(filter_size, kernel_size=(1, 1))
        # self.dropout3 = Dropout(dropout_r)

    def call(self, inputs, training, **kwargs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.act_func(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        # x = self.dropout2(x, training=training)
        xx = self.conv3(inputs)
        x += xx
        return self.act_func(x)


class SudokuSolver(tf.keras.Model):
    def __init__(self, input_shape=(9,9,1), outsize=9, nb_convlayers=10, filter_size=256, kernel=3):
        super(SudokuSolver, self).__init__()
        self.num_cnn_layer = nb_convlayers
        self.in_layer = Conv2D(filter_size, kernel_size=(kernel, kernel), input_shape=input_shape,
                               activation='relu', padding='same')
        self.in_batchnorm = BatchNormalization()
        self.in_dropout = Dropout(0.1)

        self.cnnlayers = [CnnLayer(filter_size, kernel) for _ in range(nb_convlayers)]
        self.last_layer = Conv2D(outsize, kernel_size=(1, 1), activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = self.in_layer(inputs)
        x = self.in_batchnorm(x)
        x = self.act_func(x)
        # x = self.in_dropout(x, training=training)
        for l in range(0, self.num_cnn_layer, 2):
            x = self.cnnlayers[l](x)
            x = self.cnnlayers[l+1](x)
            x = tf.keras.layers.concatenate([x, inputs], axis=-1)
        y = self.last_layer(x)
        return y


class SudokuSolver2(tf.keras.Model):
    def __init__(self, input_shape=(9,9,1), outsize=9, nb_convlayers=7, filter_size=256, kernel=3):
        super(SudokuSolver2, self).__init__()
        self.num_cnn_layer = nb_convlayers
        self.in_layer = Conv2D(filter_size, kernel_size=(kernel, kernel), input_shape=input_shape, padding='same')
        self.act_func = tf.nn.relu
        self.in_batchnorm = BatchNormalization()
        # self.in_dropout = Dropout(0.1)

        self.cnnlayers = [CnnLayer2(filter_size, kernel) for _ in range(nb_convlayers)]
        self.last_layer = Conv2D(outsize, kernel_size=(1, 1), activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, dtype=tf.float32)
        n_m = tf.equal(inputs, 0.0)
        x = self.in_layer(inputs)
        x = self.in_batchnorm(x)
        x = self.act_func(x)
        # x = self.in_dropout(x, training=training)
        for l in range(0, self.num_cnn_layer):
            x = self.cnnlayers[l](x, training)
        x = (inputs * tf.cast(tf.math.logical_not(n_m), dtype=tf.float32) + x * tf.cast(n_m, dtype=tf.float32))
        y = self.last_layer(x)
        return y


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


epochs = 50
batch_size = 64

x_all = np.load('x_all.npy', allow_pickle=True)
y_all = np.load('y_all9.npy', allow_pickle=True)

x_t_v, x_test, y_t_v, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_t_v, y_t_v, test_size=0.1, random_state=42)
print(x_train.shape)
print(y_train.shape)

train_dataset = tf.data.Dataset.from_generator(data_generator, args=(x_train, y_train),
                                               output_signature=(tf.TensorSpec(shape=(9,9,1), dtype=tf.int64),
                                                                 tf.TensorSpec(shape=(9,9,1), dtype=tf.float32)))

val_dataset = tf.data.Dataset.from_generator(data_generator, args=(x_val, y_val),
                                               output_signature=(tf.TensorSpec(shape=(9,9,1), dtype=tf.int64),
                                                                 tf.TensorSpec(shape=(9,9,1), dtype=tf.float32)))

train_batches = make_batches(train_dataset, 20000, batch_size)
val_batches = make_batches(val_dataset, 20000, 2000)



###########
# metrics #
###########

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy_grid = tf.keras.metrics.Mean('train_correct_grid', dtype=tf.float32)
train_accuracy_all = tf.keras.metrics.Mean('train_accuracy', dtype=tf.float32)

val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
val_accuracy_grid = tf.keras.metrics.Mean('val_correct_grid', dtype=tf.float32)
val_accuracy_all = tf.keras.metrics.Mean('val_accuracy', dtype=tf.float32)

log_dir_train = "logs/solver/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
log_dir_val = "logs/solver/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/validation'
summary_writer_train = tf.summary.create_file_writer(log_dir_train)
summary_writer_validation = tf.summary.create_file_writer(log_dir_val)

checkpoint_filepath = 'models/model_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'/'

model = SudokuSolver2()
optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

for epoch in range(epochs):
    for (batch, (x, y)) in enumerate(train_batches):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = sudoku_loss(y, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        grid_acc = correct_grid(y, predictions)
        full_acc = sudoku_acc(y, predictions)
        train_accuracy_grid(grid_acc)
        train_accuracy_all(full_acc)
        if batch % 1000 == 0:
            template = 'Epoch {}, Batch {}, Loss: {}, train_correct_grid: {}, train_accuracy: {}, val_loss: {}, val_correct_grid: {}, val_accuracy: {}'
            print(template.format(epoch + 1,
                                  batch,
                                  train_loss.result(),
                                  train_accuracy_grid.result() * 100,
                                  train_accuracy_all.result() * 100,
                                  val_loss.result(),
                                  val_accuracy_grid.result() * 100,
                                  val_accuracy_all.result() * 100))

    with summary_writer_train.as_default():
        tf.summary.scalar('epoch_loss', train_loss.result(), step=epoch)
        tf.summary.scalar('epoch_correct_grid', train_accuracy_grid.result(), step=epoch)
        tf.summary.scalar('epoch_sudoku_acc', train_accuracy_all.result(), step=epoch)

    for (x, y) in val_batches:
        predictions = model(x)
        loss = sudoku_loss(y, predictions)
        val_loss(loss)
        grid_acc = correct_grid(y, predictions)
        full_acc = sudoku_acc(y, predictions)
        val_accuracy_grid(grid_acc)
        val_accuracy_all(full_acc)

    with summary_writer_validation.as_default():
        tf.summary.scalar('epoch_loss', val_loss.result(), step=epoch)
        tf.summary.scalar('epoch_correct_grid', val_accuracy_grid.result(), step=epoch)
        tf.summary.scalar('epoch_sudoku_acc', val_accuracy_all.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, train_correct_grid: {}, train_accuracy: {}, val_loss: {}, val_correct_grid: {}, val_accuracy: {}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy_grid.result() * 100,
                           train_accuracy_all.result() * 100,
                           val_loss.result(),
                           val_accuracy_grid.result()*100,
                           val_accuracy_all.result()*100))
    model.save_weights(checkpoint_filepath+"e_"+str(epoch))
    # Reset metrics every epoch
    train_loss.reset_states()
    val_loss.reset_states()
    train_accuracy_grid.reset_states()
    train_accuracy_all.reset_states()
    val_accuracy_grid.reset_states()
    val_accuracy_all.reset_states()


# model = SudokuSolver2()
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=[sudoku_acc, correct_grid])
#
#
# log_dir = "logs/solver/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# checkpoint_filepath = 'models/model_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'/e-{epoch:02d}-{val_correct_grid:.2f}'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_correct_grid',
#     save_best_only=False)
#
# x_all = np.load('x_all.npy', allow_pickle=True)
# y_all = np.load('y_all9.npy', allow_pickle=True)
#
# x_t_v, x_test, y_t_v, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
# x_train, x_val, y_train, y_val = train_test_split(x_t_v, y_t_v, test_size=0.1, random_state=42)
# print(x_train.shape)
# print(y_train.shape)
# history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val),
#                     callbacks=[tensorboard_callback, model_checkpoint_callback])


