import numpy as np
#import matplotlib.pyplot as plt

def load_file(file_name):
  f = open(file_name,'r')
  data = f.read()
  f.close()
  return data

def remove_rows(data,indices):
  [N,M] = data.shape
  new_data = []
  for i in range(N):
    if not any(i == indices):
      new_data.append(data[i,:])
  new_data = np.concatenate(new_data,axis=0)
  new_data = np.reshape(new_data,[-1,M])
  return new_data

def organise(data,att_del,row_del):
  row_data = data.split(row_del)
  header = row_data[0]
  M = len(header.split(att_del))
  N = len(row_data)
  del header

  new_data = np.zeros([N,M])
  tr_indices = []
  for i in range(N):
    row = row_data[i].split(att_del)
    try:
      temp = [np.float(col) for col in row]
      new_data[i,:] = temp
    except ValueError:
      tr_indices.append(i)
      #print("Troublesome row (%d): "%i)
      #print(row) 
    del row
  tr_indices = np.array(tr_indices)
  new_data = remove_rows(new_data,tr_indices)
  return new_data

def standardise(data,target_col):
  [N,M] = data.shape
  for col in range(M):
     if col != target_col:
       mu = np.mean(data[:,col])
       std = np.std(data[:,col])
       data[:,col] = (data[:,col]-mu)/std
  return data

#data = load_file('airfoil_self_noise.dat')
#data = organise(data,"\t","\r\n")
#data = standardise(data,data.shape[1])
#print(np.matrix(data[:10,:]))

