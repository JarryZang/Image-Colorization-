import os
import tensorflow as tf
import config as config
import numpy as np
import cv2
import dataClass as data
import datetime
from functools import partial


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
from keras import applications
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import load_model, model_from_json, Model

import keras
from keras import applications
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import load_model, model_from_json, Model

#spacial attention
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K


GRADIENT_PENALTY_WEIGHT = 10



def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):

    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    save_results_path = os.path.join(config.OUT_DIR,config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, filelist +  "_reconstructed.jpg" )
    cv2.imwrite(save_path, result)
    return result

def reconstruct_no(batchX, predictedY):

    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result

def write_log(callback, names, logs, batch_no):

    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def wasserstein_loss(y_true, y_pred):

    return tf.reduce_mean(y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):

    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


class RandomWeightedAverage(_Merge):

    def _merge_function(self, inputs):
        weights = K.random_uniform((config.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class MODEL():

    def __init__(self):

        self.img_shape_1 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
        self.img_shape_2 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 2)
        self.img_shape_3 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)


        optimizer = Adam(0.00002, 0.5)
        self.discriminator = self.discriminator()
        self.discriminator.compile(loss=wasserstein_loss,
            optimizer=optimizer)

        self.colorizationModel = self.colorization_model()
        self.colorizationModel.compile(loss=['mse'],
            optimizer=optimizer)

        img_L_3 = Input(shape= self.img_shape_3)
        img_L = Input(shape= self.img_shape_1)
        img_ab_real = Input(shape= self.img_shape_2)

        self.colorizationModel.trainable = False
        predAB = self.colorizationModel(img_L_3)
        discPredAB = self.discriminator([predAB, img_L])
        discriminator_output_from_real_samples = self.discriminator([img_ab_real, img_L])


        averaged_samples = RandomWeightedAverage()([img_ab_real,
                                            predAB])
        averaged_samples_out = self.discriminator([averaged_samples, img_L])
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'


        self.discriminator_model = Model(inputs=[img_L, img_ab_real, img_L_3],
                            outputs=[discriminator_output_from_real_samples,
                                     discPredAB,
                                     averaged_samples_out])

        self.discriminator_model.compile(optimizer=optimizer,
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])



        self.colorizationModel.trainable = True
        self.discriminator.trainable = False
        self.combined = Model(inputs=[img_L_3, img_L],
                              outputs=[ predAB,  discPredAB])
        self.combined.compile(loss=['mse', wasserstein_loss],
                            loss_weights=[1.0,  -0],
                            optimizer=optimizer) #1/300


        self.log_path= os.path.join(config.LOG_DIR,config.TEST_NAME)
        self.callback = TensorBoard(self.log_path)
        self.callback.set_model(self.combined)
        self.train_names = ['loss', 'mse_loss', 'wasserstein_loss']
        self.disc_names = ['disc_loss', 'disc_valid', 'disc_fake','disc_gp']


        self.test_loss_array = []
        self.g_loss_array = []


    def discriminator(self):

        input_ab = Input(shape=self.img_shape_2, name='ab_input')
        input_l = Input(shape=self.img_shape_1, name='l_input')
        net = keras.layers.concatenate([input_l, input_ab])
        net =  keras.layers.Conv2D(64, (4, 4), padding='same', strides=(2, 2))(net) # 112, 112, 64
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(128, (4, 4), padding='same', strides=(2, 2))(net) # 56, 56, 128
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(256, (4, 4), padding='same', strides=(2, 2))(net) # 28, 28, 256
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(512, (4, 4), padding='same', strides=(1, 1))(net) # 28, 28, 512
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(1, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28,1
        return Model([input_ab, input_l], net)




    def colorization_model(self):

        input_img = Input(shape=self.img_shape_3)
        input_img_1=input_img
        input_img_2=keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(input_img)


        # VGG16 without top layer
        # VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # model_ = Model(VGG_model.input,VGG_model.layers[-6].output)
        # model = model_(input_img)

        # VGG16 without top layers(生成器网络的上半部分）
        # VGG_model_1 = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # model_1 = Model(VGG_model_1.input, VGG_model_1.layers[-6].output)
        # input_feature_1 = model_1(input_img)
        #
        # # VGG16 without top layers(生成器网络的下半部分）
        # VGG_model_2 = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # model_2 = Model(VGG_model_2.input, VGG_model_2.layers[-8].output)
        # input_feature_2 = model_2(input_img)

        input_feature_1=keras.layers.Conv2D(64,(3,3),padding='same',strides=(1,1),activation='relu')(input_img_1)
        input_feature_12 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_1)
        input_feature_13=keras.layers.MaxPooling2D(pool_size=(2,2))(input_feature_12)

        input_feature_13 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_13)
        input_feature_14 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_13)
        input_feature_15 = keras.layers.MaxPooling2D(pool_size=(2, 2))(input_feature_14)

        input_feature_15 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_15)
        input_feature_16 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_15)
        input_feature_17= keras.layers.MaxPooling2D(pool_size=(2, 2))(input_feature_16)

        input_feature_17 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_17)
        input_feature_18 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_17)




        input_feature_2 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_img_2)
        input_feature_22 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_2)
        input_feature_23 = keras.layers.MaxPooling2D(pool_size=(2, 2))(input_feature_22)

        input_feature_23 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_23)
        input_feature_24 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_23)
        input_feature_25 = keras.layers.MaxPooling2D(pool_size=(2, 2))(input_feature_24)

        input_feature_25 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_25)
        input_feature_26 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_feature_25)


        cbam_1 = channel_attention(input_feature_18)
        cbam_2 = channel_attention(input_feature_26)

        # APA
        APA_feature = keras.layers.add([cbam_1, cbam_2])

        APA_feature = keras.layers.Conv2D(240, (1, 1), padding='same', strides=(1, 1), activation='relu')(APA_feature)
        APA_feature_1 = keras.layers.Conv2D(240, (1, 1), padding='same', strides=(1, 1), activation='sigmoid')(APA_feature)
        APA_feature_out = keras.layers.Multiply()([APA_feature_1, APA_feature])


        #上采样1
        merge_1=keras.layers.concatenate([input_feature_18,APA_feature_out])
        out_feature_1=keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(merge_1)
        out_feature_1=keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_1)
        out_feature_1 = keras.layers.UpSampling2D(size=(2, 2))(out_feature_1)

        out_feature_1 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_1)
        merge_2=keras.layers.concatenate([input_feature_16,out_feature_1])
        out_feature_2 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(merge_2)
        out_feature_2 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_2)
        out_feature_2 = keras.layers.UpSampling2D(size=(2, 2))(out_feature_2)

        out_feature_2 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_2)
        merge_3 = keras.layers.concatenate([input_feature_14, out_feature_2])
        out_feature_3 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(merge_3)
        out_feature_3 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_3)
        out_feature_3 = keras.layers.UpSampling2D(size=(2, 2))(out_feature_3)

        out_feature_3 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_3)
        merge_4 = keras.layers.concatenate([input_feature_12, out_feature_3])
        out_feature_4 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(merge_4)
        out_feature_4 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_4)
        out_feature_4=keras.layers.Conv2D(2,(3,3),padding='same',strides=(1,1),activation='relu')(out_feature_4)

        # 上采样2
        merge_5 = keras.layers.concatenate([input_feature_26, APA_feature_out])
        out_feature_5 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(merge_5)
        out_feature_5 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_5)
        out_feature_5 = keras.layers.UpSampling2D(size=(2, 2))(out_feature_5)

        out_feature_5 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_5)
        merge_6 = keras.layers.concatenate([input_feature_24, out_feature_5])
        out_feature_6 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(merge_6)
        out_feature_6 = keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_6)
        out_feature_6 = keras.layers.UpSampling2D(size=(2, 2))(out_feature_6)

        out_feature_6 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_6)
        merge_7 = keras.layers.concatenate([input_feature_22, out_feature_6])
        out_feature_7 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(merge_7)
        out_feature_7 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_7)
        out_feature_7 = keras.layers.UpSampling2D(size=(2, 2))(out_feature_7)

        out_feature_7 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_7)
        # merge_4 = keras.layers.concatenate([input_feature_12, out_feature_3])
        out_feature_8 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_7)
        out_feature_8 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_8)
        out_feature_8 = keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='relu')(out_feature_8)

        FEM_output = keras.layers.concatenate([out_feature_8, out_feature_4])



        # Milevel Features  G1上采样部分

        # midlevel_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(APA_feature_out)
        # midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
        # midlevel_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(midlevel_features)
        # midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
        #
        # # Midlevel Features  G2上采样部分
        #
        # midlevel_features2 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(cbam_2)
        # midlevel_features2 = keras.layers.BatchNormalization()(midlevel_features2)
        # midlevel_features2 = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(midlevel_features2)
        # midlevel_features2 = keras.layers.BatchNormalization()(midlevel_features2)
        #
        # # Fusion + Colorization  G1上采样后半部分
        # outputModel = keras.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(midlevel_features)
        # outputModel = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        #
        # outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
        # outputModel = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        # outputModel = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        #
        # outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
        # outputModel = keras.layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        # outputModel = keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(outputModel)
        # outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
        #

        # Fusion + Colorization  G2上采样后半部分
        # outputModel2 = keras.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(midlevel_features2)
        # outputModel2 = keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel2)
        #
        # outputModel2 = keras.layers.UpSampling2D(size=(2, 2))(outputModel2)
        # outputModel2 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel2)
        # outputModel2 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel2)
        #
        # outputModel2 = keras.layers.UpSampling2D(size=(2, 2))(outputModel2)
        # outputModel2 = keras.layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel2)
        # outputModel2 = keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(outputModel2)
        # outputModel2 = keras.layers.UpSampling2D(size=(2, 2))(outputModel2)
        #
        # FEM_output = keras.layers.concatenate([outputModel2, outputMode


        GCB_feature1 = keras.layers.Conv2D(4,(1,1),padding='same',strides=(1,1), activation='softmax')(FEM_output)
        GCB_feature12 = keras.layers.Multiply()([GCB_feature1,FEM_output])
        GCB_feature12 = keras.layers.add([GCB_feature12,GCB_feature1])

        GCB_feature2 = keras.layers.Conv2D(4, (3, 3), padding='same', strides=(1, 1), activation='softmax')(FEM_output)
        GCB_feature22 = keras.layers.Multiply()([GCB_feature2, FEM_output])
        GCB_feature22 = keras.layers.add([GCB_feature22, GCB_feature2])

        GCB_feature3 = keras.layers.Conv2D(4, (5, 5), padding='same', strides=(1, 1), activation='softmax')(FEM_output)
        GCB_feature32 = keras.layers.Multiply()([GCB_feature3, FEM_output])
        GCB_feature32 = keras.layers.add([GCB_feature32, GCB_feature3])

        GCB_feature4 = keras.layers.Conv2D(4, (7, 7), padding='same', strides=(1, 1), activation='softmax')(FEM_output)
        GCB_feature42 = keras.layers.Multiply()([GCB_feature4, FEM_output])
        GCB_feature42 = keras.layers.add([GCB_feature42, GCB_feature4])

        GCB_cat = keras.layers.concatenate([GCB_feature42,GCB_feature32,GCB_feature22,GCB_feature12])
        GCB_cat = keras.layers.Conv2D(4, (1, 1), padding='same', strides=(1, 1), activation='tanh')(GCB_cat)
        GCB_cat=keras.layers.Multiply()([FEM_output,GCB_cat])
        GCB_cat=keras.layers.Conv2D(2,(1,1),padding='same',strides=1)(GCB_cat)

        final_model = Model(input=input_img, outputs=[GCB_cat])


        # Global Features

        # global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        # global_features = keras.layers.BatchNormalization()(global_features)
        # global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        # global_features = keras.layers.BatchNormalization()(global_features)
        #
        # global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(global_features)
        # global_features = keras.layers.BatchNormalization()(global_features)
        # global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        # global_features = keras.layers.BatchNormalization()(global_features)
        #
        # global_features2 = keras.layers.Flatten()(global_features)
        # global_features2 = keras.layers.Dense(1024)(global_features2)
        # global_features2 = keras.layers.Dense(512)(global_features2)
        # global_features2 = keras.layers.Dense(256)(global_features2)
        # global_features2 = keras.layers.RepeatVector(28*28)(global_features2)
        # global_features2 = keras.layers.Reshape((28,28, 256))(global_features2)
        #
        # global_featuresClass = keras.layers.Flatten()(global_features)
        # global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        # global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        # global_featuresClass = keras.layers.Dense(1000, activation='softmax')(global_featuresClass)

        # Midlevel Features

        # midlevel_features = keras.layers.Conv2D(512, (3, 3),  padding='same', strides=(1, 1), activation='relu')(model)
        # midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
        # midlevel_features = keras.layers.Conv2D(256, (3, 3),  padding='same', strides=(1, 1), activation='relu')(midlevel_features)
        # midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
        #
        # # fusion of (VGG16 + Midlevel) + (VGG16 + Global)
        # modelFusion = keras.layers.concatenate([midlevel_features, global_features2])
        #
        # # Fusion + Colorization
        # outputModel =  keras.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(modelFusion)
        # outputModel =  keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        #
        # outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        # outputModel =  keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        # outputModel =  keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        #
        # outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        # outputModel =  keras.layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        # outputModel =  keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(outputModel)
        # outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        # final_model = Model(input=input_img, outputs = [outputModel, global_featuresClass])

        return final_model


    def train(self, data,test_data, log,sample_interval=1):

        # Create folder to save models if needed.
        save_models_path =os.path.join(config.MODEL_DIR,config.TEST_NAME)
        if not os.path.exists(save_models_path):
                os.makedirs(save_models_path)

        # Load VGG network
        # VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)

        # Real, Fake and Dummy for Discriminator
        positive_y = np.ones((config.BATCH_SIZE, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((config.BATCH_SIZE, 1), dtype=np.float32)

        # total number of batches in one epoch
        total_batch = int(data.size/config.BATCH_SIZE)

        for epoch in range(config.NUM_EPOCHS):
                for batch in range(total_batch):
                    # new batch
                    trainL, trainAB, _, original, l_img_oritList  = data.generate_batch()
                    l_3=np.tile(trainL,[1,1,1,3])

                    # GT vgg
                    # predictVGG =VGG_modelF.predict(l_3)

                    # train generator
                    g_loss =self.combined.train_on_batch([l_3, trainL],
                                                        [trainAB,  positive_y])
                    # train discriminator
                    d_loss = self.discriminator_model.train_on_batch([trainL, trainAB, l_3], [positive_y, negative_y, dummy_y])

                    # update log files
                    write_log(self.callback, self.train_names, g_loss, (epoch*total_batch+batch+1))
                    write_log(self.callback, self.disc_names, d_loss, (epoch*total_batch+batch+1))

                    if (batch)%100 ==0:
                        print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" %  ( epoch, batch,total_batch, g_loss[0], d_loss[0]))
                # save models after each epoch
                if epoch % 100 == 0:
                    # save_path = os.path.join(save_models_path, "my_model_combinedEpoch%d.h5" % epoch)
                    # self.combined.save(save_path)
                    save_path = os.path.join(save_models_path, "my_model_colorizationEpoch%d.h5" % epoch)
                    self.colorizationModel.save(save_path)
                    # save_path = os.path.join(save_models_path, "my_model_discriminatorEpoch%d.h5" % epoch)
                    # self.discriminator.save(save_path)

                # sample images after each epoch
                if epoch % 100 == 0:
                    self.sample_images(test_data,epoch)


    def sample_images(self,test_data,epoch):
        total_batch = int(test_data.size/config.BATCH_SIZE)
        for _ in range(total_batch):
                # load test data
                testL, _ ,  filelist, original, labimg_oritList  = test_data.generate_batch()

                # predict AB channels
                predAB= self.colorizationModel.predict(np.tile(testL,[1,1,1,3]))

                # print results
                for i in range(config.BATCH_SIZE):
                        originalResult =  original[i]
                        height, width, channels = originalResult.shape
                        predictedAB = cv2.resize(deprocess(predAB[i]), (width,height))
                        labimg_ori =np.expand_dims(labimg_oritList[i],axis=2)
                        predResult = reconstruct(deprocess(labimg_ori), predictedAB, "epoch"+str(epoch)+"_"+filelist[i][:-5] )



if __name__ == '__main__':

    # Create log folder if needed.
    log_path= os.path.join(config.LOG_DIR,config.TEST_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")

        print('load training data from '+ config.TRAIN_DIR)
        train_data = data.DATA(config.TRAIN_DIR)
        test_data = data.DATA(config.TEST_DIR)
        assert config.BATCH_SIZE<=train_data.size, "The batch size should be smaller or equal to the number of training images --> modify it in config.py"
        print("Train data loaded")

        print("Initiliazing Model...")
        colorizationModel = MODEL()
        print("Model Initialized!")

        print("Start training")
        colorizationModel.train(train_data,test_data, log)
