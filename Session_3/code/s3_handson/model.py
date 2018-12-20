#!/usr/bin/env python

from functools import partial

import keras
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Input, concatenate, AveragePooling2D, \
    BatchNormalization, Activation
from keras import regularizers
from keras.models import Model


class DispModel():
    def __init__(self, params):
        self.model_name = params.model_name
        self.crop_height = params.crop_height
        self.crop_width = params.crop_width

        self.create_model()
        # l1 loss
        self.l1_loss = self.define_l1loss

    def create_model(self):
        input = Input(shape=(self.crop_height, self.crop_width, 6,))

        conv1_l = Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding="same", activation='relu', use_bias=1,
                         kernel_initializer='random_normal', bias_initializer='zeros',
                         kernel_regularizer=regularizers.l2(0.0001))(input)
        conv1_b = BatchNormalization()(conv1_l)
        conv1 = Activation("relu")(conv1_b)

        conv2_l = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding="same", activation='relu', use_bias=1,
                         kernel_initializer='random_normal', bias_initializer='zeros',
                         kernel_regularizer=regularizers.l2(0.0001))(conv1)
        conv2_b = BatchNormalization()(conv2_l)
        conv2 = Activation("relu")(conv2_b)

        conv3a_l = Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv2)
        conv3a_b = BatchNormalization()(conv3a_l)
        conv3a = Activation("relu")(conv3a_b)

        conv3b_l = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv3a)
        conv3b_b = BatchNormalization()(conv3b_l)
        conv3b = Activation("relu")(conv3b_b)

        conv4a_l = Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv3b)
        conv4a_b = BatchNormalization()(conv4a_l)
        conv4a = Activation("relu")(conv4a_b)

        conv4b_l = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv4a)
        conv4b_b = BatchNormalization()(conv4b_l)
        conv4b = Activation("relu")(conv4b_b)

        conv5a_l = Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv4b)
        conv5a_b = BatchNormalization()(conv5a_l)
        conv5a = Activation("relu")(conv5a_b)

        conv5b_l = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv5a)
        conv5b_b = BatchNormalization()(conv5b_l)
        conv5b = Activation("relu")(conv5b_b)

        conv6a_l = Conv2D(filters=1024, kernel_size=3, strides=(2, 2), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv5b)
        conv6a_b = BatchNormalization()(conv6a_l)
        conv6a = Activation("relu")(conv6a_b)

        print(conv1.shape, conv2.shape, conv3a.shape, conv3b.shape, conv4a.shape, conv4b.shape, conv5a.shape,
              conv5b.shape, conv6a.shape)

        conv6b_l = Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                          kernel_initializer='random_normal', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l2(0.0001))(conv6a)
        conv6b_b = BatchNormalization()(conv6b_l)
        conv6b = Activation("relu")(conv6b_b)
        pre6 = Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                      kernel_initializer='random_normal', bias_initializer='zeros',
                      kernel_regularizer=regularizers.l2(0.0001))(conv6b)

                      
        upconv5_l = Conv2DTranspose(filters=512, kernel_size=4, strides=(2, 2), activation='linear', padding="same",
                                    use_bias=1,
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=regularizers.l2(0.0001))(conv6b)
        upconv5_b = BatchNormalization()(upconv5_l)
        upconv5 = Activation("relu")(upconv5_b)
        pre6_up = UpSampling2D(size=(2, 2))(pre6)
        merge5 = concatenate([upconv5, conv5b, pre6_up], 3)
        iconv5 = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                        kernel_initializer='random_normal', bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.0001))(merge5)
        pre5 = Conv2D(filters=1, kernel_size=3, strides=(1, 1), activation='relu', padding="same", use_bias=1,
                      kernel_initializer='random_normal', bias_initializer='zeros')(iconv5)

                      
        upconv4_l = Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), activation='linear', padding="same",
                                    use_bias=1,
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=regularizers.l2(0.0001))(iconv5)
        upconv4_b = BatchNormalization()(upconv4_l)
        upconv4 = Activation("relu")(upconv4_b)
        pre5_up = UpSampling2D(size=(2, 2))(pre5)
        merge4 = concatenate([upconv4, conv4b, pre5_up], 3)
        iconv4 = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                        kernel_initializer='random_normal', bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.0001))(merge4)
        pre4 = Conv2D(filters=1, kernel_size=3, strides=(1, 1), activation='relu', padding="same", use_bias=1,
                      kernel_initializer='random_normal', bias_initializer='zeros')(iconv4)

                      
        upconv3_l = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), activation='linear', padding="same",
                                    use_bias=1,
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=regularizers.l2(0.0001))(iconv4)
        upconv3_b = BatchNormalization()(upconv3_l)
        upconv3 = Activation("relu")(upconv3_b)
        pre4_up = UpSampling2D(size=(2, 2))(pre4)
        merge3 = concatenate([upconv3, conv3b, pre4_up], 3)
        iconv3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                        kernel_initializer='random_normal', bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.0001))(merge3)
        pre3 = Conv2D(filters=1, kernel_size=3, strides=(1, 1), activation='relu', padding="same", use_bias=1,
                      kernel_initializer='random_normal', bias_initializer='zeros')(iconv3)

                      
        upconv2_l = Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), activation='linear', padding="same",
                                    use_bias=1,
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=regularizers.l2(0.0001))(iconv3)
        upconv2_b = BatchNormalization()(upconv2_l)
        upconv2 = Activation("relu")(upconv2_b)
        pre3_up = UpSampling2D(size=(2, 2))(pre3)
        merge2 = concatenate([upconv2, conv2, pre3_up], 3)
        iconv2 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                        kernel_initializer='random_normal', bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.0001))(merge2)
        pre2 = Conv2D(filters=1, kernel_size=3, strides=(1, 1), activation='relu', padding="same", use_bias=1,
                      kernel_initializer='random_normal', bias_initializer='zeros')(iconv2)

                      
        upconv1_l = Conv2DTranspose(filters=32, kernel_size=4, strides=(2, 2), activation='linear', padding="same",
                                    use_bias=1,
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=regularizers.l2(0.0001))(iconv2)
        upconv1_b = BatchNormalization()(upconv1_l)
        upconv1 = Activation("relu")(upconv1_b)
        pre2_up = UpSampling2D(size=(2, 2))(pre2)
        merge1 = concatenate([upconv1, conv1, pre2_up], 3)
        iconv1 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same", activation='relu', use_bias=1,
                        kernel_initializer='random_normal', bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.0001))(merge1)
        pre1 = Conv2D(filters=1, kernel_size=3, strides=(1, 1), activation='relu', padding="same", use_bias=1,
                      kernel_initializer='random_normal', bias_initializer='zeros')(iconv1)

                      
        upconv0_l = Conv2DTranspose(filters=16, kernel_size=4, strides=(2, 2), activation='linear', padding="same",
                                    use_bias=1,
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=regularizers.l2(0.0001))(iconv1)
        upconv0_b = BatchNormalization()(upconv0_l)
        upconv0 = Activation("relu")(upconv0_b)
        pre1_up = UpSampling2D(size=(2, 2))(pre1)
        merge0 = concatenate([upconv0, input, pre1_up], 3)
        pre0 = Conv2D(filters=1, kernel_size=5, strides=(1, 1), activation='relu', padding="same", use_bias=1,
                      kernel_initializer='random_normal', bias_initializer='zeros')(merge0)
        final_output = pre0
        # print(final_output.shape)
        
        self.model = Model(inputs=input, outputs=final_output)

        if self.model_name == 'deep_supervision':
            self.output_2 = UpSampling2D(size=(2, 2))(pre1)
            self.output_4 = UpSampling2D(size=(4, 4))(pre2)
            self.output_8 = UpSampling2D(size=(8, 8))(pre3)
            self.output_16 = UpSampling2D(size=(16, 16))(pre4)
            self.output_32 = UpSampling2D(size=(32, 32))(pre5)
            self.output_64 = UpSampling2D(size=(64, 64))(pre6)

    def err(self, a, b):
        return K.mean(K.abs(a - b))

    """ only final output used for training """
    def l1loss_simple(self, Y_true, Y_pred):
        total_loss = self.err(Y_pred, Y_true)

        return total_loss

    """ intermediate decoder outputs are also used for training """
    def l1loss_deepsup(self, Y_true, Y_pred, output_2, output_4, output_8, output_16, output_32, output_64):
        total_loss = (self.err(Y_pred, Y_true) + 1 / 2 * self.err(output_2, Y_true) + 1 / 4 * self.err(output_4, Y_true) +
                                                1 / 8 * self.err(output_8, Y_true) + 1 / 16 * self.err(output_16, Y_true) +
                                                1 / 32 * self.err(output_32, Y_true) + 1 / 64 * self.err(output_64, Y_true))

        return total_loss

    def define_l1loss(self, a, b):
        if self.model_name == 'simple':
            return self.l1loss_simple(a, b)
        if self.model_name == 'deep_supervision':
            return self.l1loss_deepsup(a, b, self.output_2, self.output_4, self.output_8, self.output_16, self.output_32, self.output_64)
