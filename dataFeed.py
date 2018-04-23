import numpy as np
import os
import re
import tensorflow as tf
from config import *
import pandas as pd

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


class DataSet:
  def __init__(self, data, label, rawData, dtype=dtypes.float32):
    self._data = data
    self._label = label
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._total_batches = data.shape[0]
    self._rawData = rawData

  @property
  def data(self):
    return self._data
  
  @property
  def rawData(self):
    return self._rawData        

  @property
  def label(self):
    return self._label

  @property
  def total_batches(self):
    return self._total_batches

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    start = self._index_in_epoch

    # first epoch shuffle
    if self._epochs_completed==0 and start==0 and shuffle:
      perm0 = np.arange(self._total_batches)
      np.random.shuffle(perm0)
      self._data = self.data[perm0]
      self._label = self.label[perm0]

    # next epoch
    if start+batch_size <= self._total_batches:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._data[start:end], self._label[start:end]

    # if the epoch is ending
    else:
      self._epochs_completed += 1
      # store what is left of this epoch
      batches_left = self._total_batches - start
      data_left = self._data[start:self._total_batches]
      label_left = self._label[start:self._total_batches]

      # shuffle for new epoch
      if shuffle:
        perm = np.arange(self._total_batches)
        np.random.shuffle(perm)
        self._data = self._data[perm]
        self._label = self._label[perm]

      # start next epoch
      start = 0
      self._index_in_epoch = batch_size - batches_left
      end = self._index_in_epoch
      data_new = self._data[start:end]
      label_new = self._label[start:end]
      return np.concatenate((data_left, data_new), axis=0), np.concatenate((label_left, label_new), axis=0)


class Base():
  def __init__(self, train, test):
    self._train = train
    self._test = test

  @property
  def train(self):
    return self._train

  @property
  def test(self):
    return self._test



def loadData(length=1000,normalize=True, offset=0):
  rawData = pd.read_csv(dataPath)
  close = rawData.Close[offset:offset+length+historyLength].copy().reset_index(drop=True)
  dataSet = rawData.iloc[offset: offset+length+historyLength, 2:2+column].values  
  history = np.zeros([length, historyLength*column ])
  label = np.zeros([length,2])
  print(history.shape)
  
  for i in range(historyLength,  length + historyLength) :
    batch = [dataSet[i-historyLength:i, 0:column]]
    batch = np.reshape(batch, [-1])
    #history = np.concatenate((history, batch), axis=0)    
    np.copyto(history[i - historyLength], batch)
    
    #Lable here
    entryPrice = close[i]
    lbl = [0.0, 1.0]
    for futureK in range(i+1, length + historyLength):
        if(close[futureK] - entryPrice > stopWin):
            lbl = [1.0, 0.0]
            break
        elif(close[futureK] - entryPrice < - stopLost):
            lbl = [0.0, 1.0]
            break
    #if(close[i-1+futureLength] - close[i-1] > 10):
    #  lbl = [[1.0, 0.0]]
    #else:
    #  lbl = [[0.0, 1.0]]
    #label = np.concatenate((label, lbl), axis=0)
    np.copyto(label[i-historyLength],lbl)
  
        
  #Normalize?
  # normalize the data
  if(normalize):    
    history_ = np.zeros(history.shape)
    for i in range(history.shape[0]):
      min = np.min(history[i], axis=0)
      max = np.max(history[i], axis=0)
      history_[i] = (history[i]-min)/(max-min)
    history = history_


  # selecting 1/(train_test_ratio+1) for testing, and train_test_ratio/(train_test_ratio+1) for training
  train_test_idx = int(trainTestRatio *label.shape[0])
  trainHistoryData = history[:train_test_idx]
  trainLabels = label[:train_test_idx]
  trainRawData = dataSet[:train_test_idx]    
  testHistoryData = history[train_test_idx:]
  testLabels = label[train_test_idx:]
  testRawData = dataSet[train_test_idx:]

  train = DataSet(trainHistoryData, trainLabels, trainRawData)
  test = DataSet(testHistoryData, testLabels, testRawData)
  base = Base(train=train, test=test)
  return base