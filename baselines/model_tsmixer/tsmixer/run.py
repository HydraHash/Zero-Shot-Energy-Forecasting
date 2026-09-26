# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train and evaluate models for time series forecasting with local visualization."""

import argparse
import glob
import logging
import os
import time

# Attempt to import original loader; if it expects GCS, we patch the global path variables
import data_loader 
from data_loader import TSFDataLoader
import models

from models import tsmixer
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# 🛠️ WINDOWS GCS PATCH: Force the data_loader's cloud path prefix to target your local data folder directly
if hasattr(data_loader, 'BURI'):
    # Adjust this path if your ETT or Weather files are stored somewhere else
    data_loader.BURI = './datasets/' 
if hasattr(data_loader, 'GCS_DATA_PATH'):
    data_loader.GCS_DATA_PATH = './datasets/'

def parse_args():
  """Parse the arguments for experiment configuration."""

  parser = argparse.ArgumentParser(
      description='TSMixer for Time Series Forecasting'
  )

  # basic config
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument(
      '--model',
      type=str,
      default='tsmixer',
      help='model name, options: [tsmixer, tsmixer_rev_in]',
  )

  # data loader
  parser.add_argument(
      '--data',
      type=str,
      default='ETTh1', # Changing default to your known working file
      choices=[
          'electricity', 'exchange_rate', 'national_illness', 'traffic',
          'weather', 'ETTm1', 'ETTm2', 'ETTh1', 'ETTh2',
      ],
      help='data name',
  )
  parser.add_argument(
      '--feature_type',
      type=str,
      default='M',
      choices=['S', 'M', 'MS'],
      help=(
          'forecasting task, options:[M, S, MS]; M:multivariate predict'
          ' multivariate, S:univariate predict univariate, MS:multivariate'
          ' predict univariate'
      ),
  )
  parser.add_argument(
      '--target', type=str, default='OT', help='target feature in S or MS task'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./checkpoints/',
      help='location of model checkpoints',
  )
  parser.add_argument(
      '--delete_checkpoint',
      action='store_true',
      help='delete checkpoints after the experiment',
  )

  # forecasting task
  parser.add_argument(
      '--seq_len', type=int, default=96, help='input sequence length (shrunk for testing)'
  )
  parser.add_argument(
      '--pred_len', type=int, default=24, help='prediction sequence length'
  )

  # model hyperparameter
  parser.add_argument(
      '--n_block', type=int, default=2, help='number of blocks'
  )
  parser.add_argument(
      '--ff_dim', type=int, default=64, help='fully-connected feature dimension'
  )
  parser.add_argument(
      '--dropout', type=float, default=0.05, help='dropout rate'
  )
  parser.add_argument(
      '--norm_type', type=str, default='B', choices=['L', 'B'], help='LayerNorm or BatchNorm'
  )
  parser.add_argument(
      '--activation', type=str, default='relu', choices=['relu', 'gelu'], help='Activation function'
  )

  # optimization
  parser.add_argument(
      '--num_workers', type=int, default=0, help='data loader num workers'
  )
  parser.add_argument(
      '--train_epochs', type=int, default=1, help='train epochs (1 for fast check)'
  )
  parser.add_argument(
      '--batch_size', type=int, default=8, help='batch size of input data'
  )
  parser.add_argument(
      '--learning_rate', type=float, default=0.001, help='optimizer learning rate'
  )
  parser.add_argument(
      '--patience', type=int, default=3, help='number of epochs to early stop'
  )

  # save results
  parser.add_argument(
      '--result_path', default='result.csv', help='path to save result'
  )

  args = parser.parse_args()
  tf.keras.utils.set_random_seed(args.seed)
  return args


def main():
  args = parse_args()
  
  exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}'

  # load datasets
  print("📂 Loading local data channels...")
  try:
      data_loader = TSFDataLoader(
          args.data,
          args.batch_size,
          args.seq_len,
          args.pred_len,
          args.feature_type,
          args.target,
      )
  except Exception as e:
      print(f"⚠️ Native data loader failed parsing GCS paths: {e}")
      print("💡 Action: Ensure your data directory mappings in 'data_loader.py' point to actual local folders.")
      raise e

  train_data = data_loader.get_train()
  val_data = data_loader.get_val()
  test_data = data_loader.get_test()

  # train model
  print("🏗️ Initializing TSMixer Architecture...")
  if args.model == 'tsmixer':
    build_model = tsmixer.build_model
  else:
     build_model = getattr(models, args.model).build_model
  model = build_model(
      input_shape=(args.seq_len, data_loader.n_feature),
      pred_len=args.pred_len,
      norm_type=args.norm_type,
      activation=args.activation,
      dropout=args.dropout,
      n_block=args.n_block,
      ff_dim=args.ff_dim,
      target_slice=data_loader.target_slice,
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
  model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
  
  os.makedirs(args.checkpoint_dir, exist_ok=True)
  checkpoint_path = os.path.join(args.checkpoint_dir, f'{exp_id}_best.weights.h5')
  
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      verbose=1,
      save_best_only=True,
      save_weights_only=True,
  )
  early_stop_callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', patience=args.patience
  )
  
  print("⚡ Executing model execution train loop...")
  history = model.fit(
      train_data,
      epochs=args.train_epochs,
      validation_data=val_data,
      callbacks=[checkpoint_callback, early_stop_callback],
  )

  # evaluate best model
  model.load_weights(checkpoint_path)
  print("🧪 Evaluating performance on test dataset...")
  test_result = model.evaluate(test_data)
  print(f"Test Evaluation - MSE: {test_result[0]:.4f} | MAE: {test_result[1]:.4f}")

  # =============================================================
  # 📈 VISUALIZATION ADDITION BLOCK
  # =============================================================
  print("\n🔮 Extracting test batch for plotting generation...")
  
  # Take the first batch from the test set iterator
  for batch_x, batch_y in test_data.take(1):
      # Generate actual predictions
      batch_preds = model.predict(batch_x)
      
      # Extract sample index 0 from the batch
      # If multivariate, let's look at the target dimension axis
      sample_x = batch_x[0].numpy()
      sample_y = batch_y[0].numpy()
      sample_pred = batch_preds[0]
      
      # If multivariate, grab the specific target channel slice (usually the last feature)
      target_channel = -1 
      
      history_series = sample_x[:, target_channel]
      actual_future = sample_y[:, target_channel]
      predicted_future = sample_pred[:, target_channel] if len(sample_pred.shape) > 1 else sample_pred
      
      plt.figure(figsize=(11, 4))
      plt.plot(range(args.seq_len), history_series, label="Historical Lookback Context", color="royalblue")
      plt.plot(range(args.seq_len, args.seq_len + args.pred_len), actual_future, label="Ground Truth (Actuals)", color="green")
      plt.plot(range(args.seq_len, args.seq_len + args.pred_len), predicted_future, label="TSMixer Forecast", color="crimson", linestyle="--")
      
      plt.axvline(x=args.seq_len, color='black', linestyle=':', alpha=0.5)
      plt.title(f"TSMixer Local Baseline Verification Forecast ({args.data})")
      plt.xlabel("Time Steps (Hourly)")
      plt.ylabel("Value Dimensions")
      plt.legend()
      plt.tight_layout()
      plt.show()
      break # Process only one sample batch visualization window

if __name__ == '__main__':
  main()