import numpy as np
import os
import urllib
import gzip
import pickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir, k =10 , add_noise = False, add_v =None,sample_size= None):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if sample_size is not None:
        images = images[0:sample_size,:]
        labels = labels[0:sample_size]
        
    images_chunks = np.split(images,k)
    labels_chunks = np.split(labels,k)
    
    if add_noise:
        for j in range(k):
            images_chunks[j] = images_chunks[j] + add_v[j]
    def get_epoch():        
        for i in range(k):
            rng_state = np.random.get_state()
            np.random.shuffle(images_chunks[i])
            np.random.set_state(rng_state)
            np.random.shuffle(labels_chunks[i])
            #np_ind = np.random.permutation(np.shape(images)[0]/k)
            #images_chunks[i] = images_chunks[i][np_ind,:]
            #labels_chunks[i] = labels_chunks[i][np_ind]
        for i in range(int(len(images) / (k*batch_size))):
            yield [(images_chunks[j][i*batch_size:(i+1)*batch_size,:], labels_chunks[j][i*batch_size:(i+1)*batch_size]) for j in range(k)]

    return get_epoch


def load(batch_size, test_batch_size, data_dir, k=10, add_noise = False, add_v = None, sample_size=None):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,k=k, add_noise = add_noise, add_v = add_v,sample_size=sample_size), 
        cifar_generator(['test_batch'], test_batch_size, data_dir, k=1,sample_size=np.minimum(sample_size,10000))
    )