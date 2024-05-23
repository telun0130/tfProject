import os
# solution for：oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import *
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Concatenate, Add, Multiply, UpSampling2D
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import HeNormal, RandomNormal

class Block():
  def VGG16_Block1(self, img_input, kernal_size):
    x = Conv2D(kernal_size, (3, 3),
               activation='relu',  # "sigmoid"
               padding='same',
               kernel_initializer=RandomNormal(stddev=0.02),
               )(img_input)

    x = Conv2D(kernal_size, (3, 3),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(stddev=0.02),
               )(x)

    sc = x

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return sc, x

  def VGG16_Block2(self, img_input, kernal_size, The5th_Block=False):
    x = Conv2D(kernal_size, (3, 3),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(stddev=0.02),
               )(img_input)

    x = Conv2D(kernal_size, (3, 3),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(stddev=0.02),
               )(x)

    x = Conv2D(kernal_size, (3, 3),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(stddev=0.02),
               )(x)

    if The5th_Block == False:
      sc = x
      x = MaxPooling2D((2, 2), strides=(2, 2))(x)
      return sc, x
    else:
      return x

  def _Up_Conv(self, input, ch_out):
    upsampX = UpSampling2D(size=(2, 2))(input)
    x = Conv2D(ch_out, kernel_size=3, padding='same')(upsampX)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

  def _Rconv(self, input, ch_out, t):
    for i in range(t):
      if i == 0:
        x = Conv2D(ch_out, kernel_size=3, padding='same')(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
      x = Conv2D(ch_out, kernel_size=3, padding='same')(input + x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
    return x

  def _R2block1(self, input, ch_out, t):
    x1 = Conv2D(ch_out, kernel_size=1, padding='same')(input)
    x2 = self._Rconv(x1, ch_out, t)
    x2 = self._Rconv(x2, ch_out, t)
    x_sc = x1 + x2
    x3 = MaxPooling2D(pool_size=(2, 2))(x_sc)
    return x_sc, x3

  def _R2block2(self, input, ch_out, t):
    x1 = Conv2D(ch_out, kernel_size=1, padding='same')(input)
    x2 = self._Rconv(x1, ch_out, t)
    x2 = self._Rconv(x2, ch_out, t)
    return x1 + x2

  def attention_gate(self, input_tensor, gating_signal, inter_channel):
    theta_x = Conv2D(inter_channel, (2, 2), padding='same')(input_tensor)
    phi_g = Conv2D(inter_channel, (1, 1), padding='same')(gating_signal)

    concat = Add()([theta_x, phi_g])
    act = ReLU()(concat)
    psi = Conv2D(1, (1, 1), padding='same')(act)
    sig = sigmoid(psi)
    #  upsample_psi = UpSampling2D(size=(2, 2))(sigmoid)

    # 乘上原始輸入張量以聚焦到更重要的部分
    output = Multiply()([input_tensor, sig])
    return output

# 主要模型，繼承了Block類來構築模型
class ModelSet(Block):
  def VGG16(self, img_input):
    sc1, block1 = self.VGG16_Block1(img_input, 64)  # sc1 : 512,512,3 -> 512,512,64 ,block1 : 512,512,64 -> 256,256,64
    sc2, block2 = self.VGG16_Block1(block1, 128)  # sc2 : 256,256,64 -> 256,256,128 ,block2 : 256,256,128 -> 128,128,128
    sc3, block3 = self.VGG16_Block2(block2, 256,
                               The5th_Block=False)  # sc3 : 128,128,128 -> 128,128,256 ,block3 : 128,128,256 -> 64,64,256
    sc4, block4 = self.VGG16_Block2(block3, 512,
                               The5th_Block=False)  # sc4 : 64,64,256 -> 64,64,512 ,block4 : 64,64,512 -> 32,32,512
    block5 = self.VGG16_Block2(block4, 512, The5th_Block=True)  # block5 = 32,32,512 -> 32,32,512

    return sc1, sc2, sc3, sc4, block5

  def Unet(self, num_classes=2, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):  # 黑白CHANNELS = 1 , RGB = 3
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    sc1, sc2, sc3, sc4, block5 = self.VGG16(inputs)

    channels = [64, 128, 256, 512]

    block5_up = UpSampling2D(size=(2, 2))(block5)  # 32, 32, 512 -> 64, 64, 512

    sc4_gating = self.attention_gate(sc4, block5_up, 512)

    block4_up = Concatenate(axis=3)([sc4_gating, block5_up])  # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024

    block4_up = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block4_up)
    block4_up = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block4_up)  # 64, 64, 1024 -> 64, 64, 512

    block4_up = UpSampling2D(size=(2, 2))(block4_up)  # 64, 64, 512 -> 128, 128, 512

    sc3_gating = self.attention_gate(sc3, block4_up, 256)

    block3_up = Concatenate(axis=3)([sc3_gating, block4_up])  # 128, 128, 256 + 128, 128, 512 -> 128, 128, 768

    block3_up = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block3_up)
    block3_up = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block3_up)  # 128, 128, 768 -> 128, 128, 256

    block3_up = UpSampling2D(size=(2, 2))(block3_up)  # 128, 128, 256 -> 256, 256, 256

    sc2_gating = self.attention_gate(sc2, block3_up, 128)

    block2_up = Concatenate(axis=3)([sc2_gating, block3_up])  # 256, 256, 256 + 256, 256, 128 -> 256, 256, 384

    block2_up = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block2_up)
    block2_up = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block2_up)  # 256, 256, 384 -> 256, 256, 128

    block2_up = UpSampling2D(size=(2, 2))(block2_up)  # 256, 256, 128 -> 512, 512, 128

    sc1_gating = self.attention_gate(sc1, block2_up, 64)

    block1_up = Concatenate(axis=3)([sc1_gating, block2_up])  # 512, 512, 128 + 512, 512, 64 -> 512, 512, 192

    block1_up = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block1_up)
    block1_up = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
      block1_up)  # 512, 512, 192 -> 512, 512, 64

    output = Conv2D(num_classes, 1, activation="sigmoid")(
      block1_up)  # 512, 512, 64 -> 512, 512, num_classes #softmax or sigmoid

    model = Model(inputs=inputs, outputs=output)
    return model

  def R2U(self, Inputs, ch_out):
    # Ch_in -> 64
    X1_sc, X1 = self._R2block1(Inputs, 64, t=2)
    # 64 -> 128
    X2_sc, X2 = self._R2block1(X1, 128, t=2)
    # 128 -> 256
    X3_sc, X3 = self._R2block1(X2, 256, t=2)
    # 256 -> 512
    X4_sc, X4 = self._R2block1(X3, 512, t=2)
    # 512 -> 1024
    X5 = self._R2block2(X4, 1024, t=2)

    # 1024 -> 512
    d5 = self._Up_Conv(X5, 512)
    d5 = Concatenate()([d5, X4_sc])
    d5 = self._R2block2(d5, 512, t=2)
    # 512 -> 256
    d4 = self._Up_Conv(d5, 256)
    d4 = Concatenate()([d4, X3_sc])
    d4 = self._R2block2(d4, 256, t=2)
    # 256 -> 128
    d3 = self._Up_Conv(d4, 128)
    d3 = Concatenate()([d3, X2_sc])
    d3 = self._R2block2(d3, 128, t=2)
    # 128 -> 64
    d2 = self._Up_Conv(d3, 64)
    d2 = Concatenate()([d2, X1_sc])
    d2 = self._R2block2(d2, 64, t=2)
    # 64 -> Ch_out
    d1 = Conv2D(ch_out, kernel_size=1, padding='same')(d2)
    output = sigmoid(d1)

    return output

class ModelCreator(ModelSet):
  def __init__(self):
    pass
  # 輸出模型
  def OutputModel(self, Modelname, input_channel, output_channel):
    if Modelname == 'R2U-Net':
      Inputs = Input(shape=(256, 256, input_channel))
      outputs = self.R2U(Inputs, output_channel)
      model = Model(inputs=Inputs, outputs=outputs)
      model.compile(optimizer='Adam', loss='binary_crossentropy',
                      metrics=[metrics.IoU(num_classes=2, target_class_ids=[1]), 'accuracy'])
      model.summary()
      return [True, model]
    elif Modelname == 'U-Net':
      model = self.Unet(num_classes=2, IMG_HEIGHT=512, IMG_WIDTH=512, IMG_CHANNELS=1)  # 黑白IMG_CHANNELS=1,彩色IMG_CHANNELS=3
      model.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=[metrics.IoU(num_classes=2, target_class_ids=[1])])  # loss = "Dice" "sparse_categorical_crossentropy不用one_hot encoding" "categorical_crossentropy 要做one-hot encoding"
      model.summary()
      return [True, model]
    else:
      return [False, "Model unexists"]

  # 選擇模型並輸出
  def BuildModel(self, ch_in, num_class, option): # option為模型選項 (U-Net, R2U-Net....)
    model = self.OutputModel(Modelname=option, input_channel=ch_in, output_channel=num_class)
    return model

if __name__ == '__main__':
  try:
    modelcaller = ModelCreator()
    modelcaller.BuildModel(ch_in=1, num_class=2, option='U-Net')
  except Exception as E:
    print(E)
