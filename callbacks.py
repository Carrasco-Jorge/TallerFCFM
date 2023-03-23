from .configuration import get_config
from tensorflow.keras.callbacks import EarlyStopping,\
                            ReduceLROnPlateau, ModelCheckpoint,\
                            TensorBoard, Callback
import tensorflow as tf
from tensorflow.summary import create_file_writer
from tensorflow.summary import image as tfImage
import matplotlib.pyplot as plt
import numpy as np
import datetime
import io

class PlotBatch(Callback):
  def __init__(self, val, logdir, num_samples=5):
    batch_size = get_config().get('batch_size')
    self.batch_size = batch_size
    self.val = val
    self.num_samples = np.min([num_samples,batch_size])
    self.file_writer = create_file_writer(logdir)
    
  def on_epoch_end(self, epoch, epoch_logs):
    # Plot examples
    img, msk = next(iter(self.val.data))
    pred = self.model.predict(img,batch_size=self.batch_size)

    fig, ax = plt.subplots(nrows=3, ncols=self.num_samples, figsize=(4*self.num_samples,12))
    for i in range(self.num_samples):
      ax[0][i].imshow(img[i], vmin=0.0,vmax=1.0)
      ax[0][i].axis("off")

      ax[1][i].imshow(msk[i], cmap='gray',vmin=0.0,vmax=1.0)
      ax[1][i].axis("off")

      ax[2][i].imshow(pred[i], cmap='gray',vmin=0.0,vmax=1.0)
      ax[2][i].axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)

    with self.file_writer.as_default():
      tfImage('Val. images', plot, step=epoch)


def get_callbacks(val):
  earlyStopping = EarlyStopping(
    monitor='val_dice_coef',
    patience=2,
    verbose=1,
    mode='max'
  )

  reduceLR = ReduceLROnPlateau(
    monitor='val_dice_coef',
    factor=0.1,
    patience=1,
    verbose=1,
    mode='max'
  )

  name = 'model_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  checkpoint = ModelCheckpoint(
    f'./logs/checkpoints/{name},h5',
    monitor='val_dice_coef',
    save_best_only=True,
    mode='max'
  )

  tensorboard = TensorBoard(
    log_dir=f'./logs/fit/{name}',
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
  )

  return [earlyStopping, reduceLR, checkpoint, tensorboard, PlotBatch(val,f'./logs/fit/{name}')], name






