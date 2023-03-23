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
from tqdm import tqdm

class PlotBatch(Callback):
  def __init__(self, val, logdir, num_samples=5):
    batch_size = get_config().get('batch_size')
    self.batch_size = batch_size
    self.val = val
    self.num_samples = np.min([num_samples,batch_size])
    self.file_writer = create_file_writer(logdir)
    
  def on_epoch_end(self, epoch, epoch_logs=None):
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
      tfImage('Imágenes val.', plot, step=epoch)


class PlotMetrics(Callback):
  def __init__(self, val, logdir, num_ticks=30):
    self.batch_size = val.size
    self.val = val
    self.val.data = self.val.data.rebatch(self.val.size)
    self.num_ticks = num_ticks
    self.file_writer = create_file_writer(logdir)
  
  def on_epoch_end(self, epoch, epoch_logs=None):
    eps =  0.00001
    conf_levels = np.linspace(0.0, 1.0, self.num_ticks)
    max_f1 = 0
    acc = []
    p   = []
    r   = []
    f1  = []
    fpr = []
    opt_conf_matrix = np.zeros((2,2))

    flatten = tf.keras.layers.Flatten()

    for conf in tqdm(conf_levels, total=self.num_ticks):
      # Get predictions over validation dataset
      #  TP FN
      #  FP TN
      conf_matrix = np.zeros((2,2))

      for img, msk in self.val.data:
        pred = self.model.predict(img)
        # Flatten tensors
        shp = (self.batch_size*256**2)
        msk  = tf.reshape(flatten(msk), shp)
        pred = tf.reshape(tf.cast(flatten(pred) > conf, tf.float32), shp)
        conf_m = tf.math.confusion_matrix(msk, pred).numpy()
        conf_matrix += conf_m
        break

      # Compute confusion matrix
      tp = conf_matrix[0][0]
      fn = conf_matrix[0][1]
      fp = conf_matrix[1][0]
      tn = conf_matrix[1][1]

      # Compute metrics
      acc.append( (tp + tn) / np.sum(conf_matrix) )
      p.append( tp / (tp + fp + eps) )
      r.append( tp / (tp + fn + eps) )
      f1.append( 2*(p[-1] * r[-1]) / (p[-1] + r[-1] + eps) )
      if f1[-1] > max_f1:
        max_f1 = f1[-1]
        opt_conf_matrix = conf_matrix.copy()
      fpr.append( fp / (fp + tn + eps) )

    # Plot metrics
    #  Plot acc, p, r, fpr
    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(conf_levels, acc, label='Acc')
    ax.plot(conf_levels, p, label='P')
    ax.plot(conf_levels, r, label='R')
    ax.plot(conf_levels, fpr, label='FPR')
    plt.legend()
    plt.title('Métricas de matriz de confusión')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)

    with self.file_writer.as_default():
      tfImage('Métricas mat. conf.', plot, step=epoch)
    #  Plot F1 curve
    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(conf_levels, f1, label='F1')
    plt.legend()
    plt.title('Curva F1')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)

    with self.file_writer.as_default():
      tfImage('Curva F1', plot, step=epoch)
    #  Plot PR curve
    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(r, p, label='PR')
    plt.legend()
    plt.title('Curva PR')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)

    with self.file_writer.as_default():
      tfImage('Curva PR', plot, step=epoch)
    #  Plot ROC curve
    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(fpr, r, label='ROC')
    plt.legend()
    plt.title('Curva ROC')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)

    with self.file_writer.as_default():
      tfImage('Curva ROC', plot, step=epoch)
    #  Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(opt_conf_matrix)
    for row in [0,1]:
      for col in [0,1]:
        ax.text(col, row, opt_conf_matrix[row, col], fontsize=20)
    plt.title('Matriz de confusión')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)

    with self.file_writer.as_default():
      tfImage('Matriz conf.', plot, step=epoch)

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

  return [
    earlyStopping, 
    reduceLR, 
    checkpoint, 
    tensorboard, 
    PlotBatch(val,f'./logs/fit/{name}'),
    PlotMetrics(val,f'./logs/fit/{name}')
    ], name






