"""
Implement and compare the performance of a neural network using different activation
functions (Sigmoid, ReLU, Tanh) on the MNIST dataset. Analyze how each activation
function affects the training process and classification accuracy.
Tasks:
● Load and preprocess the MNIST dataset.
● Implement neural networks using Sigmoid, ReLU, and Tanh activation functions.
● Train and evaluate each network.
● Compare training times, convergence, and classification accuracy.

"""
import tensorflow as tf
import time

(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()

activations=["sigmoid","relu","tanh"]

for activation in activations:
    model=tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128,activation=activation),
        tf.keras.layers.Dense(10,activation="softmax")
        ])
    model.compile(loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    start=time.time()
    history=model.fit(X_train,y_train,epochs=5)
    end=time.time()
    loss,accuracy=model.evaluate(X_test,y_test)
    print("\nActivation: ",activation,"\nAccuracy: ",accuracy,"\nTraining time: ",end-start)
# import tensorflow as tf
# import time

# # Load MNIST dataset
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Normalize pixel values
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # Flatten images
# X_train = X_train.reshape(-1, 784)
# X_test = X_test.reshape(-1, 784)

# # Activation functions
# activations = ["sigmoid", "relu", "tanh"]

# for activation in activations:

#     print("\nActivation:", activation)

#     # Create neural network
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(128, activation=activation,
#                               input_shape=(784,)),
#         tf.keras.layers.Dense(64, activation=activation),
#         tf.keras.layers.Dense(10, activation="softmax")
#     ])

#     # Compile model
#     model.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     # Measure training time
#     start = time.time()

#     history = model.fit(
#         X_train,
#         y_train,
#         epochs=5,
#         batch_size=128,
#         validation_split=0.1,
#         verbose=1
#     )

#     end = time.time()

#     # Evaluate
#     loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

#     print("Training Time:", round(end - start, 2), "seconds")
#     print("Test Accuracy:", round(accuracy, 4))