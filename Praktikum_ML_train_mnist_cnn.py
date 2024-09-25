# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:37:34 2022
@author: Kirko
MNIST-Zahlen laden, CNN-Model trainieren, image vorhersagen, Ausgabe plotten
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler,\
    ReduceLROnPlateau, ProgbarLogger
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,\
    InputLayer, BatchNormalization, RandomRotation,\
    RandomBrightness, RandomContrast, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, metrics
from sklearn.metrics import ConfusionMatrixDisplay     


# normalize all train_images and test_imnages for best cnn
def normalize_train_test(train_img, test_img):
    erwart_train = np.mean(train_img)
    standard_train = np.std(train_img)
    train_img = (train_img-erwart_train)/standard_train
    erwart_test = np.mean(test_img)
    standard_test = np.std(test_img)
    test_img = (test_img-erwart_test)/standard_test
    return train_img, test_img

# adjust the learning rate for best cnn
def scheduler(epoch, l_rate):
    if epoch < 4:
        return l_rate
    return 0.001

# # load train and test data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalize the images between 0 and 1.
train_images = train_images / 255# - 0.5
test_images = test_images / 255 #- 0.5

# display one train image
# plt.imshow(train_images[0],cmap="binary")
# oneImg = train_images[0]

# normalize all images
train_images, test_images = normalize_train_test(train_images, test_images)

# Reshape the images from (28,28) to (28, 28, 1) because keras requires three dim 
print(train_images[0].shape)
train_images = np.expand_dims(train_images, axis=3)
test_images  = np.expand_dims(test_images, axis=3)

print("train_images.shape: " + str(train_images.shape))
print("train_labels.shape: " + str(train_labels.shape))
# print(test_images[0].shape)

def plot_predict(label):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_example, cmap="gray")
    plt.title("Testimage")
    plt.text(9, 29.5, "testlabel: "+str(label), fontsize="large")
    plt.axis("off")
    plt.xlabel("{:n} ".format(np.argmax(img_predict[0])))
    plt.subplot(1,2,2)
    plt.bar(range(10), img_predict[0], color="blue")
    plt.grid(False)
    plt.title("Barplot")
    plt.xticks(range(10))
    plt.ylim([0, 1])

train = 0
if train == 1:
    # define variables
    NUM_FILTER = 16
    FILTER_SIZE = 3
    EPOCHS = 10

    # #define modelarchitecture with one Convolution and one Pooling
    # model = Sequential([
    #   Conv2D(16,
    #           FILTER_SIZE,
    #           input_shape = (28, 28, 1),
    #           # strides = (3,3),
    #           activation = "relu",
    #           padding ="same",
    #           name="conv1"),

    #   MaxPooling2D(pool_size = 2,
    #                 name = "pool1"),

    #   Flatten(name="flatten"),

    #   Dense(10,
    #         activation = 'softmax',
    #         name = "dense1"),
    # ])

    # architecture from tensorflownet
    # model = Sequential([
    #   Conv2D(16,
    #           5,
    #           input_shape = (28, 28, 1),
    #           activation = "relu",
    #           padding ="same",
    #           name="conv1"),

    #   MaxPooling2D(pool_size = 2,
    #                 name = "pool1"),

    #   Conv2D(32,
    #           5,
    #           activation = "relu",
    #           padding ="same",
    #           name="conv2"),

    #   MaxPooling2D(pool_size = 2,
    #                 name = "pool2"),

    #   Flatten(name="flatten"),

    #   Dense(128,
    #         activation = 'relu',
    #         name = "dense1"),

    #   Dense(10,
    #         activation = 'softmax',
    #         name = "dense2"),
    # ])

    
    # data_augmentation = Sequential([
    #     InputLayer(input_shape=(28,28,1)),
    #     RandomFlip("horizontal_and_vertical"),
    #     RandomRotation(0.4),
    #     ])
    # data_augmentation.build()

    # # define CNN with the least parameters
    model = Sequential([
        InputLayer(input_shape=(28, 28, 1)),
        # RandomFlip("horizontal_and_vertical"),
        # RandomRotation(0.1),
        # RandomBrightness(0.5),
        # RandomContrast(0.5),
        
        Conv2D(6,
              5,
              name = "conv_1",
              # input_shape = (28, 28, 1),
              # strides = (2,1),
              activation = "relu",
               # dilation_rate = (3,3),
              padding ="same"),

        MaxPooling2D(pool_size = 2,
                      name = "pool_1"),
        
        # Dropout(0.05),

        Conv2D(5,
              2,
              name = "conv_2",
              activation = "relu",
              # dilation_rate = (2,2),
              padding ="same"),

        MaxPooling2D(pool_size = 2,
                      name = "pool_2"),

        Conv2D(4,
                3,
                name = "conv_3",
                activation = "relu",
                padding ="same"),

        MaxPooling2D(pool_size = 2,
                      name = "pool_3"),

        Flatten(name = "flatten"),

        Dense(10,
              activation = 'softmax',
              name = "dense"),
    ])

    # show network architecture
    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.1,
                                  patience = 2,
                                  min_lr = 0.0001)

    # compile the model
    model.compile(
      optimizer = optimizers.Adam(learning_rate=1e-2),  #Adam(learning_rate=1e-2)
                  loss      = 'categorical_crossentropy',
                  metrics   = ['accuracy'],
    )

    callbacks = [
        TensorBoard(
            log_dir         = "log_dir",
            histogram_freq  = 1,
            embeddings_freq = 1,
        ),
        LearningRateScheduler(scheduler),
        # reduce_lr,
        # ProgbarLogger(count_mode='samples')
    ]

    #train the model
    H = model.fit(
      train_images,
      to_categorical(train_labels),
      batch_size = 256,
      epochs=EPOCHS,
      validation_data=(test_images, to_categorical(test_labels)),
      callbacks = callbacks,
      )

    model.save("PML_MNIST_CNN_KIRKO_acc_loeschen")
    print(H.params)

    def plot_train():
        # plot the model training history
        N = EPOCHS
        # plt.style.use("seaborn-v0_8")
        plt.style.use("default")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc", color = "red")
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss", color = "red")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc", color = "green")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss", color = "green")
        plt.title("Bounding Box Regression Loss on Training Set")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
    plot_train()

else:
    # model = load_model("PML_MNIST_CNN_KIRKO_acc_0.9818") #
    # model = load_model("PML_MNIST_CNN_KIRKO_acc_schlecht") #
    model = load_model("PML_MNIST_CNN_KIRKO_tensorflownet") #
    # model = load_model("PML_MNIST_CNN_AlexanderBjoernJulian") #
    # model = load_model("PML_MNIST_CNN_Liebert") #
    # model = load_model("PML_MNIST_CNN_NitschkeLibov") #
    # model = load_model("PML_MNIST_CNN_Patrick_JanuszV4") #
    # model = load_model("PML_MNIST_CNN_AlbrechtYazdifar") #
    # model = load_model("PML_MNIST_CNN_KarimCreutzburg") #
    # model = load_model("PML_MNIST_CNN_HoenaKrause") #
    # model = load_model("PML_MNIST_CNN_Liemt_Weigel") #
    model.summary()
    one_or_all = 1

    if one_or_all == 1:
        
        
        counter = 2  # Beispiel auswählen
        img_example = test_images[counter]
        test_label  = test_labels[counter]
        test_image  = np.expand_dims(img_example, axis=0)
        img_predict = model.predict(test_image)

        for i, x in enumerate(img_predict[0]):
            print(i, "=", x*100)
        # print("img_predict: ", img_predict)
        print("img_label: ", test_labels[counter])
        # plot_predict(test_label)

    else:
        pre_label = model.predict(test_images)
        pre_label = np.argmax(pre_label, axis=1)

        cm = ConfusionMatrixDisplay.from_predictions(test_labels,
                                                      pre_label,
                                                      cmap="gist_heat",
                                                      colorbar=True
                                                      )
        cm.ax_.set_title("Confusion Matrix MNIST-Ziffern")
        plt.grid(False) #Gitter ausschalten
        calc_acc = sum(pre_label == test_labels)/len(pre_label)
        print(calc_acc)
         

def plot_from_gui(image):
    predict = model.predict(image)
    return predict
