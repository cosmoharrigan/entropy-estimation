# Estimation of Mutual Information for Image Classification
# Author: Cosmo Harrigan
#
# Creates a binarized version of the MNIST dataset and
# estimates the mutual information between the class label
# and pixel-level features or the top-k pixel feature vectors
#
# See paper for details: 
# http://www.machineintelligence.org/papers/mutual-information-mnist.pdf

import numpy as np
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
from sklearn.preprocessing import binarize
from skimage.transform import downscale_local_mean, rescale
from sklearn.preprocessing import LabelBinarizer

def entropy(p):
  """Returns the entropy of a probability mass function p"""
  total = 0
  for i in range(len(p)):
    if p[i] != 0:
      total += -p[i]*np.log2(p[i])
  return total

def entropy_vectorized(p_vector):
  """Returns a vector with the entropies of each random variable, given a 
  vector of random variables"""
  entropy = np.sum(-np.multiply(p_vector, np.ma.log2(p_vector)), axis=1)
  entropy = np.ma.fix_invalid(entropy, fill_value=0)
  entropy = np.array(entropy)
  return entropy

def show(image_vector, colorbar=False, hot=False):
  n = np.int(np.sqrt(image_vector.shape[0]))
  if hot:
    ax = plt.imshow(image_vector.reshape(n, n), cmap='hot')
  else:
    ax = plt.imshow(image_vector.reshape(n, n))
  ax.axes.grid(None)
  plt.colorbar(ax)
  return plt

mnist = fetch_openml('mnist_784', cache=False)

binary_mnist_data = binarize(mnist.data)
binary_mnist_target = mnist.target.astype(np.int)

N = 5000

binary_mnist_test_data = binary_mnist_data[N:2*N]
binary_mnist_test_target = binary_mnist_target[N:2*N]

binary_mnist_data = binary_mnist_data[0:N]
binary_mnist_target = binary_mnist_target[0:N]

digit_samples = []
for i in range(10):
  digit_samples.append(binary_mnist_data[np.where(binary_mnist_target == i)])
plt = show(digit_samples[0][0]);
plt.savefig('sample.png')

# H(Y)
p_Y = np.unique(binary_mnist_target, return_counts=True)[1] / binary_mnist_target.shape[0]

print('Entropy of class label:', entropy(p_Y))

def P_Y_given_X(x):
  P_Y_given_X_equals_x = np.zeros((784, 10))
  for i in range(784):
    count_total = binary_mnist_data[binary_mnist_data[:, i] == x].shape[0]

    count_per_class = np.zeros((10))
    counts_found = np.unique(binary_mnist_target[binary_mnist_data[:, i] == x], return_counts=True)
    if len(counts_found[0]) > 0:    
      if len(counts_found[0] == 1):
        count_per_class[counts_found[0]] = counts_found[1]
      else:
        for digit, digit_count in counts_found[0], counts_found[1]:
          count_per_class[digit] = digit_count

    if count_total == 0:
      probability = np.zeros((10))
    else:
      probability = count_per_class / count_total 
    P_Y_given_X_equals_x[i] = probability
  
  return P_Y_given_X_equals_x

P_Y_given_X_equals_0 = P_Y_given_X(x=0)
P_Y_given_X_equals_1 = P_Y_given_X(x=1)

P_X_equals_1 = binary_mnist_data.mean(axis=0)
P_X_equals_0 = 1 - binary_mnist_data.mean(axis=0)

I_X_Y = entropy(p_Y) - entropy_vectorized(P_Y_given_X_equals_0) * P_X_equals_0 - entropy_vectorized(P_Y_given_X_equals_1) * P_X_equals_1

# Mutual information of each pixel with the class label
plt = show(I_X_Y, colorbar=True, hot=True)
plt.xlabel("x-axis of image");
plt.ylabel("y-axis of image");
plt.savefig("single-pixel-mi.png");

print('I(X;Y) min, max, mean: ', I_X_Y.min(), I_X_Y.max(), I_X_Y.mean())

# Rank the pixels by prediction accuracy

def hard_prediction(i, pixel):
  if binary_mnist_data[i, pixel] == 1:
    return np.argmax(P_Y_given_X_equals_1[pixel])
  elif binary_mnist_data[i, pixel] == 0:
    return np.argmax(P_Y_given_X_equals_0[pixel])

pixel_prediction_accuracy = np.zeros(784)

for pixel in range(784):#best_predictors:
  correct = 0
  for i in range(N):    
    pred = hard_prediction(i, pixel)
    target = binary_mnist_target[i]

    if pred == target:
      correct += 1
      
  pixel_prediction_accuracy[pixel] = correct / N

plt = show(pixel_prediction_accuracy, colorbar=True, hot=True)
plt.xlabel("x-axis of image");
plt.ylabel("y-axis of image");
plt.savefig("single-pixel-accuracy.png");

# Take the top-k pixels by mutual information with Y, and use them as a feature vector

def top_k_pixels(k):
  top_pixel_idx = np.argpartition(I_X_Y, -k)[-k:]
  top_pixels = binary_mnist_data[:, top_pixel_idx]
  counts_top_pixels = np.unique(top_pixels, axis=0, return_counts=True)
  sorted = np.argsort(-counts_top_pixels[1])

  # H(X)
  prob_top_pixels = counts_top_pixels[1] / binary_mnist_data.shape[0]
  H_X = entropy(prob_top_pixels)
  
  # H(X|Y) = SUM of p(y) * H(X|Y=y)
  H_X_given_Y = 0
  for i in range(10):
    p_X_given_Y = np.unique(top_pixels[binary_mnist_target == i], axis=0, return_counts=True)[1] / binary_mnist_target[binary_mnist_target == i].shape[0]  
    H_X_given_Y += p_Y[i] * entropy(p_X_given_Y)
  
  # I(X; Y) = H(X) - H(X|Y)
  top_k_I_X_Y = H_X - H_X_given_Y
  
  return top_k_I_X_Y

k_range = list(range(1, 60)) + list(range(60, 100, 10)) + [100]
I_based_on_k = []
for k in k_range:
  I_based_on_k.append(top_k_pixels(k))

plt.scatter(k_range, I_based_on_k);
plt.xlabel('# of pixels');
plt.ylabel('Mutual information with target');
plt.savefig('top-k-mi.png')

# Evaluate predictive accuracy on test set

def array_to_string(array):
  array = array.astype(np.int)
  s = ""
  for i in range(array.shape[0]):
    s += str(array[i])
  return s

def prediction_top_k(k):
  # Select the k pixels with the best prediction accuracy from the training set
  top_pixel_idx = np.argpartition(pixel_prediction_accuracy, -k)[-k:]
  top_pixels = binary_mnist_data[:, top_pixel_idx]
  
  counts_top_pixels = np.unique(top_pixels, axis=0, return_counts=True)
  unique_patterns = counts_top_pixels[0]

  # Construct a classifier that predicts a class label based on the top-k pixels
  limited_dataset = binary_mnist_data[:, top_pixel_idx]
  predictor = np.zeros((unique_patterns.shape[0]), dtype=np.int)

  for i in range(unique_patterns.shape[0]):
    pattern = unique_patterns[i]
    
    per_digit = np.unique(binary_mnist_target[(limited_dataset == pattern).all(axis=1).nonzero()[0]], return_counts=True)
    counts_per_digit = per_digit[1]
    digits = per_digit[0]
    
    p_Y_given_X = counts_per_digit / binary_mnist_target[(limited_dataset == pattern).all(axis=1).nonzero()[0]].shape[0]

    predictor[i] = digits[np.argmax(p_Y_given_X)].astype(np.int)

  classifier = {}
  for i in range(unique_patterns.shape[0]):
    classifier[array_to_string(unique_patterns[i])] = predictor[i]

  return top_pixel_idx, unique_patterns, classifier

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
accuracy = []

for k in k_values:
  print("k=", k)
  
  top_pixel_idx, unique_patterns, classifier = prediction_top_k(k)
  
  num_correct = 0
  not_found = 0
  
  for i in range(binary_mnist_test_target.shape[0]):
    pattern_found = binary_mnist_test_data[i, top_pixel_idx]

    if pattern_found is not None:
      pattern_string = array_to_string(pattern_found)
      
      try:
        if classifier[pattern_string] == binary_mnist_test_target[i]:
          num_correct += 1
      except KeyError:
        # Pattern not present in training data
        not_found += 1
        pass
    
  result = num_correct / binary_mnist_test_target.shape[0]
  print("Accuracy: ", result)
  print("Not found %: ", not_found / N)
  accuracy.append(result)

plt.scatter(k_values, accuracy);
plt.xlabel('# of pixels');
plt.ylabel('Prediction accuracy');
plt.savefig('test-set-prediction-accuracy.png');
