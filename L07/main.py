import sys
from keras import Sequential, Input
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def split_by_space(arr: str):
    return list(map(float, arr.split()))


def random_stuff_idk():
    file = open("./data/tryt_48.dat", "r")
    data = list(map(split_by_space, file.read().splitlines()))
    test_data = list(map(lambda a: a[0:len(a)-1], data))
    ideal_data = list(map(lambda a: a[len(a)-1], data))
    X_train, X_test, y_train, y_test = train_test_split(test_data, ideal_data, test_size=0.3)
    model = Sequential(name="feed-forward")  # Model
    model.add(Input(shape=(48,), name='Input-Layer'))  # Input Layer - need to speicfy the shape of inputs
    model.add(Dense(5, activation='tansig', name='Hidden-Layer'))  # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(Dense(1, activation='logsig', name='Output-Layer'))

    model.compile(optimizer='traindx',  # default='rmsprop', an algorithm to be used in backpropagation
                  loss='binary_crossentropy',
                  metrics=['Accuracy', 'Precision', 'Recall'],
                  loss_weights=None,
                  weighted_metrics=None,
                  run_eagerly=None,
                  steps_per_execution=None
                  )

    model.fit(X_train,  # input data
              y_train,  # target data
              batch_size=10,  # Number of samples per gradient update. If unspecified, batch_size will default to 32.
              epochs=3,
              # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
              verbose='auto',
              # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
              callbacks=None,  # default=None, list of callbacks to apply during training. See tf.keras.callbacks
              validation_split=0.2,
              # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
              # validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
              shuffle=True,
              # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
              class_weight={0: 0.3, 1: 0.7},
              # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
              sample_weight=None,
              # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
              initial_epoch=0,
              # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
              steps_per_epoch=None,
              # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
              validation_steps=None,
              # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
              validation_batch_size=None,
              # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
              validation_freq=3,
              # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
              max_queue_size=10,
              # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
              workers=1,
              # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
              use_multiprocessing=False,
              # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
              )
    # Predict class labels on training data
    pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
    # Predict class labels on a test data
    pred_labels_te = (model.predict(X_test) > 0.5).astype(int)

    ##### Step 7 - Model Performance Summary
    print("")
    print('-------------------- Model Summary --------------------')
    model.summary()  # print model summary
    print("")
    print('-------------------- Weights and Biases --------------------')
    for layer in model.layers:
        print("Layer: ", layer.name)  # print layer name
        print("  --Kernels (Weights): ", layer.get_weights()[0])  # weights
        print("  --Biases: ", layer.get_weights()[1])  # biases

    print("")
    print('---------- Evaluation on Training Data ----------')
    print(classification_report(y_train, pred_labels_tr))
    print("")

    print('---------- Evaluation on Test Data ----------')
    print(classification_report(y_test, pred_labels_te))
    print("")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '1' in sys.argv:
            random_stuff_idk()
