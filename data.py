import cv2 as cv
import numpy as np
import os
import tensorflow as tf

def imgProcess(imgroot, img_name):
  img = cv.imread(os.path.join(imgroot, img_name))
  img = cv.resize(img, (256, 256))
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  img = cv.medianBlur(img, ksize=5)
  kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
  img = cv.erode(img, kernel)
  img = img / 255.
  img = np.array(img)
  img = np.expand_dims(img, axis=-1)
  return img

def maskProcess(maskroot, mask_name):
  mask = cv.imread(os.path.join(maskroot, mask_name))
  mask = cv.resize(mask, (256, 256))
  mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
  mask = np.array(mask)
  mask[mask > 0] = 1
  new_mask = np.zeros((256, 256, 2), dtype=np.uint8)
  new_mask[:, :, 0] = (mask == 0).astype(np.uint8)
  new_mask[:, :, 1] = (mask == 1).astype(np.uint8)
  return new_mask

def DScreate(data_root, label_root):
  data_file = os.listdir(data_root)
  label_file = os.listdir(label_root)
  data_file.sort()
  label_file.sort()

  Data = []
  Label = []
  for data_filename, label_filename in zip(data_file, label_file):
    data = imgProcess(data_root, data_filename)
    label = maskProcess(label_root, label_filename)
    Data.append(data)
    Label.append(label)
  Datas = tf.data.Dataset.from_tensor_slices(Data)
  Labels = tf.data.Dataset.from_tensor_slices(Label)
  dataset = tf.data.Dataset.zip((Datas, Labels))
  dataset = dataset.batch(batch_size=1)
  return dataset

if __name__ == '__main__':
  Train_set = DScreate('train_1000/img', 'train_1000/mask')
  print("build success")