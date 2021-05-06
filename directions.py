import numpy as np

def get_generators_weights(Gen, resolutions=[2,4,16,32,64,128,256,512,1024]):

  weights = []
  for d in resolutions:
    res = f'{d}x{d}'
    
    for layer_name in Gen.vars.keys():
      if res in layer_name and "ToRGB" not in layer_name and "mod_weight" in layer_name:
        # print (layer_name)
        layer = Gen.vars[layer_name]
        w = Gen.components.synthesis.get_var(layer)
        weights.append(w)	
  return weights
  
def normalize_vectors(weight):
  return weight / np.linalg.norm(weight, axis=0, keepdims=True)

def eigenvectors(weight):
  # return eigen_values and eigen_vectors
  eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
  return eigen_values, eigen_vectors.T 

def centered_covariance(weight):
  return weight.dot(weight.T)
