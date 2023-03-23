from tensorflow.keras.callbacks import EarlyStopping,\
                            ReduceLROnPlateau, ModelCheckpoint,\
                            TensorBoard
import datetime

def get_callbacks():
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
    histogram_freq=1
  )

  return [earlyStopping, reduceLR, checkpoint, tensorboard], name






