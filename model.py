import tensroflow as tf
from .configuration import get_config
from .metrics import dice_coef
from .loss import BceDiceLoss
from .utils import Timer

def init_model(model):
  config = get_config
  model.compile(
      optimizer=config.get('opt')(learning_rate=config.get('lr')),
      # loss='binary_crossentropy',
      loss=BceDiceLoss(weights=[0.7,0.3]),
      weighted_metrics=[
          dice_coef,
          tf.keras.metrics.BinaryIoU(target_class_ids=[1])          
      ]
  )
  
  model.summary()

  return model

def run_experiment(model, train, val, test):
  config = get_config()
  timer = Timer()
  timer.begin()
  model.fit(
      train.data,
      epochs=config.get('epochs'),
      steps_per_epoch=train.size//config.get('batch_size'),
      validation_data=val.data,
      validation_steps=val.size//config.get('batch_size')
  )
  timer.end()

  model.evaluate(test.data, steps=test.size)

