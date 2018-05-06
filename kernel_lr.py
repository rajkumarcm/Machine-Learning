import preprocess as pre
import numpy as np
from matplotlib import pyplot as plt
import pca
from kmeans import *

data = pre.load_file('Data/Airfoil/airfoil_self_noise.dat')
data = pre.organise(data, "\t", "\r\n")
data = pre.standardise(data, data.shape[1])
PCA = 1
new_dim = 4
N_val_data = 300
K = 10
k_means = 1

t = np.reshape(data[N_val_data:, 5], [-1, 1])
val_t = np.reshape(data[:N_val_data, 5], [-1, 1])
val_data = data[:N_val_data, :5]
data = data[N_val_data:, :5]

lr = 1e-1
fig_title = "Before applying PCA"

if PCA:
    # Perform PCA
    data = pca.project_data(data, new_dim)
    val_data = pca.project_data(val_data, new_dim)
    fig_title = "After applying PCA"

[N, M] = data.shape

centroid,width,_ = kmeans(data,K)
new_data = np.zeros([N,K])
new_val_data = np.zeros([N_val_data,K])

# Create the data matrix with by using RBF
for i in range(N):
    for k in range(K):
        temp = data[i] - centroid[k]
        new_data[i,k] = np.exp(-np.dot(temp,temp.T)/(2*width[k]))

for i in range(N_val_data):
    for k in range(K):
        temp_val = val_data[i] - centroid[k]
        new_val_data[i,k] = np.exp(-np.dot(temp_val,temp_val.T)/(2*width[k]))


data = np.copy(new_data)
val_data = np.copy(new_val_data)
del new_data
del new_val_data
M = data.shape[1]


# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(new_data[:,0],new_data[:,1],new_data[:,2],marker='d',c='r')
W = np.zeros([M, 1])
b = 0
# try:
#     W = np.load('Data/Airfoil/random_weight.npy')
#     b = np.load('Data/Airfoil/random_bias.npy')
# except Exception:
#     W = np.random.random([M, 1])
#     b = np.random.rand()
#     np.save('Data/Airfoil/random_weight.npy', W)
#     np.save('Data/Airfoil/random_bias.npy', b)
W = np.random.random([M,1])
if PCA and not k_means:
    W = W[:new_dim, :]
# W = np.dot(np.dot(np.linalg.inv(np.dot(data.T,data)),data.T),t)
data = data.T  # Examples are arranged in columns [features,N]
val_data = val_data.T
b = np.random.rand()
epochs = 600
loss = np.zeros([epochs])
val_loss = np.zeros([epochs])

fig = plt.figure()
for epoch in range(epochs):
    if epoch % 1000 == 0:
        lr /= 10
    # Obtain the output
    y = np.dot(W.T, data).T + b
    val_y = np.dot(W.T, val_data).T + b

    sse = np.dot((t - y).T, (t - y))
    val_sse = np.dot((val_t - val_y).T, (val_t - val_y))

    loss[epoch] = sse / N
    val_loss[epoch] = val_sse / N_val_data
    var = sse / N
    # log likelihood
    ll = (-N / 2) * (np.log(2 * np.pi)) \
         - (N * np.log(np.sqrt(var))) \
         - (sse / (2 * var))
    plt.scatter(epoch, loss[epoch], c='b', label="Training Error")
    plt.scatter(epoch, val_loss[epoch], c='r', label="Validation Error")
    if epoch == 0:
        plt.title("Kernel based Linear Regression")
        plt.legend(loc="upper right")
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
    plt.pause(1e-10)

    # Gradient Descent

    W_grad = np.zeros([M, 1])
    B_grad = 0
    for i in range(N):
        err = (t[i] - y[i])
        # W_grad += err * np.reshape(data[:,i],[-1,1])
        W_grad += data[:, i].reshape([-1, 1]) * (y[i] - t[i]).reshape([])
        B_grad += y[i] - t[i]

    W_grad /= N
    B_grad /= N

    W -= lr * W_grad
    b -= lr * B_grad

    print("Epoch: %d, Loss: %.3f, Val Loss: %.3f, Log-Likelihood: %.3f"
          %(epoch, loss[epoch], val_loss[epoch], ll))

y = np.dot(W.T, data).T + b
plt.figure()
plt.plot(range(len(t)), t, '-b')
plt.plot(range(len(y)), y, '-r')
plt.show()
# --------------------------
# plt.figure()
# plt.plot(range(epochs),loss,'-r')
# plt.show()
