import tensorflow as tf
import numpy as np

device = 'cpu'

def changeDevice(devName):
  """Changes active device to either CPU or GPU"""
  global device
  device = devName

def dot(_x,_y):
  """Dot product"""
  with tf.device('/{}:0'.format(device)):
    x=tf.constant(_x)
    y=tf.constant(_y)
    result=tf.reduce_sum(tf.multiply(x,y))
  out=0
  sess=tf.Session()
  out=sess.run(result)
  sess.close()
  return out

def cross(_x,_y):
  """Cross product"""
  with tf.device('/{}:0'.format(device)):
    x=tf.constant(_x)
    y=tf.constant(_y)
    result=tf.cross(x,y)
  out=0
  sess=tf.Session()
  out=sess.run(result)
  sess.close()
  return out

def add(_x,_y):
  """Vector/Matrix addition"""
  with tf.device('/{}:0'.format(device)):
    x=tf.constant(_x)
    y=tf.constant(_y)
    result=tf.add(x,y)
  out=0
  sess=tf.Session()
  out=sess.run(result)
  sess.close()
  return out

def matmul(_x,_y):
  """Matrix product"""
  with tf.device('/{}:0'.format(device)):
    x=tf.constant(_x)
    y=tf.constant(_y)
    result=tf.matmul(x,y)
  out=0
  sess=tf.Session()
  out=sess.run(result)
  sess.close()
  return out
