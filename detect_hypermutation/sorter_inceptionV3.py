import os
import re
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import multi_gpu_model
import numpy as np
from random import randint, random
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

class Sorter():
    img_rows = 300
    img_cols = 300
    channels = 3

    def __init__(self, classes=[], train_dir="", validation_dir="",
                 save_weights_path="./weights.h5",
                 finetuning_weights_path="", img_size=(300, 300),
                 color_randomize_options=None,
                 n_gpus=1,
                 ealry_stopping_options=None,
                 n_epochs=100):
        if not hasattr(classes, "__iter__"):
            raise ValueError("classes must be iterable")
        elif len(classes) < 1:
            raise ValueError("classes is empty")

        self.__preloaded_model = None
        self.classes = classes
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.save_weights_path = save_weights_path
        self.finetuning_weights_path = finetuning_weights_path
        self.img_rows = img_size[0]
        self.img_cols = img_size[1]
        self.batch_size = 50 * n_gpus
        self.n_gpus = n_gpus
        self.color_randomize_options = color_randomize_options
        self.original_model = None
        self.ealry_stopping_options=ealry_stopping_options
        self.n_epochs = n_epochs

    def train(self):
        if self.train_dir == "" or self.validation_dir == "":
            raise ValueError("train_dir and/or validation_dir is empty")

        img_rows = self.img_rows
        img_cols = self.img_cols
        classes = self.classes
        n_classes = len(classes)
        channels = self.channels
        batch_size = self.batch_size
        n_epochs = self.n_epochs

        n_train_samples = self._count_files(self.train_dir)
        n_val_samples = self._count_files(self.validation_dir)

        print("training samples  : {}".format(n_train_samples))
        print("validation samples: {}".format(n_val_samples))

        trainImageGenerator = ImageDataGenerator(
            rescale = 1 / 255,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=self._preprocess,
        )

        train_generator = trainImageGenerator.flow_from_directory(
            directory=self.train_dir,
            target_size=(img_rows, img_cols),
            color_mode='rgb',
            classes=classes,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
        )

        validationImageGenerator = ImageDataGenerator(
            rescale = 1 / 255,
        )

        validation_generator = validationImageGenerator.flow_from_directory(
            directory=self.validation_dir,
            target_size=(img_rows, img_cols),
            color_mode='rgb',
            classes=classes,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )

        if self.finetuning_weights_path:
            model = self.model(weights_path=self.finetuning_weights_path)
        else:
            model = self.model()

        # define callbacks
        s = self.ealry_stopping_options
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10 if s is None else s['patience'],
            verbose=0,
            mode="auto",
        )

        # define TensorBoard
        tensorboard_callback = TensorBoard(
            log_dir="./logs",
        )

        callbacks = [
            early_stopping_callback,
            tensorboard_callback
        ]

        # tuning
        history = model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            steps_per_epoch=n_train_samples / batch_size,
            validation_steps=n_val_samples / batch_size,
            epochs=n_epochs,
            callbacks=callbacks,
        )

        if self.n_gpus > 1:
            self.original_model.save_weights(self.save_weights_path)
        else:
            model.save_weights(self.save_weights_path)

    def evaluate(self, validation_dirs=None):
        if self.validation_dir == "" and validation_dirs is None:
            raise ValueError("validation_dir is empty")

        img_rows = self.img_rows
        img_cols = self.img_cols
        classes = self.classes
        n_classes = len(classes)
        channels = self.channels
        batch_size = self.batch_size
        n_epochs = self.n_epochs

        if self.finetuning_weights_path:
            model = self.model(weights_path=self.finetuning_weights_path)
        else:
            model = self.model()

        if validation_dirs is None:
            n_val_samples = self._count_files(self.validation_dir)
            print(n_val_samples)

            validationImageGenerator = ImageDataGenerator(
                rescale = 1 / 255,
            )

            validation_generator = validationImageGenerator.flow_from_directory(
                directory=self.validation_dir,
                target_size=(img_rows, img_cols),
                color_mode='rgb',
                classes=classes,
                class_mode='categorical',
                batch_size=batch_size,
                shuffle=True
            )

            # tuning
            history = model.evaluate_generator(
                validation_generator,
                steps=n_val_samples
            )

            print("accuracy: {}".format(history[1]))

        else:
            for val_dir in validation_dirs:
                n_val_samples = self._count_files(val_dir)
                print(val_dir)
                print(n_val_samples)

                validationImageGenerator = ImageDataGenerator(
                    rescale = 1 / 255,
                )

                validation_generator = validationImageGenerator.flow_from_directory(
                    directory=val_dir,
                    target_size=(img_rows, img_cols),
                    color_mode='rgb',
                    classes=classes,
                    class_mode='categorical',
                    batch_size=batch_size,
                    shuffle=True
                )

                # tuning
                history = model.evaluate_generator(
                    validation_generator,
                    steps=n_val_samples
                )

                print("accuracy: {}".format(history[1]))


    def preload_model(self):
        self.__preloaded_model = self.model(self.finetuning_weights_path)

        return self.__preloaded_model

    def detect(self, filename=""):
        if self.__preloaded_model is None:
            model = self.preload_model()
        else:
            model = self.__preloaded_model

        classes = self.classes

        img = load_img(filename, target_size=(300, 300))
        x = img_to_array(img)
        x = x / 255

        x = np.asarray([x])
        pred = model.predict(x)[0]

        top = pred.argsort()[::-1][0]

        return classes[top]


    def model(self, weights_path=None, n_freeze=20):
        img_rows = self.img_rows
        img_cols = self.img_cols
        classes = self.classes
        n_classes = len(classes)

        # model
        inception_v3 = InceptionV3(include_top=False, weights="imagenet", input_shape=(img_rows, img_cols, 3))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=inception_v3.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(n_classes, activation='softmax'))

        # attach
        model = Model(inputs=[inception_v3.input], outputs=[top_model(inception_v3.output)])

        # load weights
        if weights_path is not None:
            model.load_weights(weights_path)

        # freeze weights
        # https://github.com/danielvarga/keras-finetuning/blob/master/train.py
        how_many = n_freeze
        for layer in model.layers[:how_many]:
            layer.trainable = False
        for layer in model.layers[how_many:]:
            layer.trainable = True

        # multi-gpu
        if self.n_gpus > 1:
            self.original_model = model
            model = multi_gpu_model(self.original_model, gpus=self.n_gpus)

        # compile
        lr = 1e-4 / 2
        momentum = 0.9
        model.compile(loss='categorical_crossentropy',
                                  optimizer=optimizers.SGD(lr=lr, momentum=momentum),
                                    metrics=['accuracy'])


        return model

    def _count_files(self, base_dir):
        def _is_imagefile(filename):
            if re.match(r'.+\.(png|jpg|jpeg|bmp|tiff)$', filename):
                return True
            else:
                return False

        count = 0
        for category in self.classes:
            files = os.listdir(os.path.join(base_dir, category))
            files = [f for f in files if _is_imagefile(f)]
            count += len(files)

        return count

    def _preprocess(self, tensor):
        if self.color_randomize_options is not None:
            o = self.color_randomize_options
            h_range = o['h']
            s_range = o['s']
            v_range = o['v']
        else:
            return tensor

        tensor_hsv = rgb_to_hsv(tensor)
        tt = tensor_hsv.T
        h = tt[0]
        s = tt[1]
        v = tt[2]

        h = np.clip(h + (random() * (h_range * 2) - h_range), 0, 1)
        s = np.clip(s + (random() * (s_range * 2) - s_range), 0, 1)
        v = np.clip(v + randint(-v_range, v_range), 0, 255)

        rs = self.img_rows
        cs = self.img_cols
        tensor_hsv = np.concatenate(
            (
                h.reshape(1, rs, cs),
                s.reshape(1, rs, cs),
                v.reshape(1, rs, cs),
            ),
            axis=0).T

        result = hsv_to_rgb(tensor_hsv)
        return result



if __name__ == '__main__':
    main()
