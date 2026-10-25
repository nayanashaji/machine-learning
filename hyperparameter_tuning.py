"""
Implement and perform hyperparameter tuning for a neural network on the Fashion
MNIST dataset. Experiment with different learning rates, batch sizes, and epochs, and
discuss the impact on model performance.
Tasks:
● Load and preprocess the Fashion MNIST dataset.
● Experiment with different hyperparameters (learning rate, batch size, epochs).
● Train and evaluate the network.
● Discuss how hyperparameter choices affect model performance.

"""
import tensorflow as tf
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

(X_train,y_train),(X_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()

X_train=X_train/255
X_test=X_test/255

X_train=X_train.reshape(-1,784)
X_test=X_test.reshape(-1,784)

learning_rates=[0.001,0.01]
batch_sizes=[32,64]
epochs_list=[5,10]

for lr in learning_rates:
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            model=MLPClassifier(
                hidden_layer_sizes=(10,),
                learning_rate_init=lr,
                batch_size=batch_size,
                max_iter=epochs
            )
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            print(
                f"LR={lr}, Batch={batch_size}, Epochs={epochs}, "
                f"Accuracy={accuracy_score(y_test,y_pred)}"
            )
# import tensorflow as tf
# import time

# # Load Fashion MNIST dataset
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# # Normalize pixel values
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # Flatten images
# X_train = X_train.reshape(-1, 784)
# X_test = X_test.reshape(-1, 784)

# # Hyperparameters
# learning_rates = [0.001, 0.01]
# batch_sizes = [32, 64]
# epochs_list = [5, 10]

# # Experiment with different hyperparameters
# for lr in learning_rates:
#     for batch_size in batch_sizes:
#         for epochs in epochs_list:

#             print("\nLearning Rate:", lr)
#             print("Batch Size:", batch_size)
#             print("Epochs:", epochs)

#             # Create model
#             model = tf.keras.Sequential([
#                 tf.keras.layers.Dense(128, activation="relu",
#                                       input_shape=(784,)),
#                 tf.keras.layers.Dense(64, activation="relu"),
#                 tf.keras.layers.Dense(10, activation="softmax")
#             ])

#             # Compile model
#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#                 loss="sparse_categorical_crossentropy",
#                 metrics=["accuracy"]
#             )

#             # Train
#             start = time.time()

#             model.fit(
#                 X_train,
#                 y_train,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 validation_split=0.1,
#                 verbose=0
#             )

#             training_time = time.time() - start

#             # Evaluate
#             loss, accuracy = model.evaluate(
#                 X_test, y_test, verbose=0
#             )

#             print("Test Accuracy:", accuracy)
#             print("Training Time:", round(training_time, 2), "seconds")