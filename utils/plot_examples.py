import matplotlib.pyplot as plt
import numpy as np

def plot_sample(X,noisy_X,mean,variance,name,dataset,index):
    for i in index:
        if mean is not None:
            M = mean[i].data.cpu().numpy().reshape(X[0].shape)
            M = np.swapaxes(np.swapaxes(M, 0, 2), 0, 1)[:, :, 0]
            plt.figure()
            plt.imshow(M)
            plt.colorbar()
            plt.savefig('./visualization/{}_{}_{}_mean_map.png'.format(dataset,name,i))
        if variance is not None:
            V = variance[i].data.cpu().numpy().reshape(X[0].shape)
            V = np.swapaxes(np.swapaxes(V, 0, 2), 0, 1)[:, :, 0]
            plt.figure()
            plt.imshow(V)
            plt.colorbar()
            plt.savefig('./visualization/{}_{}_{}_variance_map.png'.format(dataset,name,i))

        N=noisy_X[i].data.cpu().numpy().reshape(X[0].shape)
        N = np.swapaxes(np.swapaxes(N, 0, 2), 0, 1)
        plt.figure()
        plt.imshow(N)
        plt.colorbar()
        plt.savefig('./visualization/{}_{}_{}_noisy_input.png'.format(dataset,name,i))

        O=X[i].data.cpu().numpy().reshape(X[0].shape)
        O = np.swapaxes(np.swapaxes(O, 0, 2), 0, 1)
        plt.figure()
        plt.imshow(O)
        plt.colorbar()
        plt.savefig('./visualization/{}_{}_{}_input.png'.format(dataset,name,i))