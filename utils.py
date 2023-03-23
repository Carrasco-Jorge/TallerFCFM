from .configuration import get_config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def viz_images(ds):
  config = get_config()
  # image_batch, mask_batch, sw_batch = next(iter(val))
  image_batch, mask_batch = next(iter(ds.data))
  rows = config.get('batch_size')//2
  fig, ax = plt.subplots(ncols=rows,nrows=2,figsize=(15, 6))
  for i in range(rows):
    ax[0][i].imshow(image_batch[i].numpy().astype("float32"),vmin=0,vmax=1,cmap='gray')
    ax[1][i].imshow(mask_batch[i].numpy().astype("float32"),vmin=0,vmax=1,cmap='gray')
    plt.axis("off")
  plt.show()

class Timer():
  def begin(self):
    self.start = time.time()
  
  def end(self):
    total_time = time.time() - self.start
    minutes = int(total_time // 60)
    seconds = np.round(total_time % 60)
    
    print(f'Time: {minutes} minutes {seconds} seconds.\n')

def viz_predictions(model, ds, num_samples=3):
  config = get_config()
  # Predict
  counter = 1
  for sample in ds.data:
    img, msk = sample
    print(img.shape, msk.shape)
    y_pred = model.predict(img,batch_size=config.get('batch_size'))
    print(y_pred.shape)
    
    for counter in range(num_samples):#config.get('batch_size')):
      fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
      ax[0].imshow(tf.image.resize(img[counter], config.get("mask_shape")))
      ax[0].set_title('Image')

      ax[1].imshow(msk[counter], cmap='Blues',vmin=0.0,vmax=1.0)
      ax[1].set_title('Mask')

      ax[2].imshow(y_pred[counter], cmap='Reds',vmin=0.0,vmax=1.0)
      ax[2].set_title('Prediction')

      ax[3].imshow(tf.math.round(y_pred[counter]), cmap='Reds',vmin=0.0,vmax=1.0)
      ax[3].set_title('Binarized prediction')

      plt.show()

    break
