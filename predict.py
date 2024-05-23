import tensorflow as tf
from data import imgProcess
from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class predictor():
    def __init__(self, modelPath):
        if modelPath.endswith('.keras'):
            self.Model = tf.keras.models.load_model(modelPath)
        else:
            print('sorry, you need a keras file.')
    def pred(self, x):
        prediction = self.Model.predict(x)
        return prediction

if __name__ == '__main__':
    predict = predictor("Model/R2U-Net.keras")
    predict.Model.summary()

    x = imgProcess("train_1000/img", "CFD_041.png")
    Datas = tf.data.Dataset.from_tensor_slices([x])
    Datas = Datas.batch(batch_size=1)
    output = predict.pred(Datas)*255
    outMask = np.transpose(output[0], (2, 0, 1)).astype(np.uint8)
    image = Image.fromarray(outMask[1])
    image.save('output_image_2.png')  # 保存圖片
    image.show()