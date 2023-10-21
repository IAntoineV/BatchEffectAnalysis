import pickle
import numpy as np
import matplotlib.pyplot as plt


def check_dict(dic):
    """Function to check what are the elements of an dictionary

    Args:
        dic (dict)
    """
    for key in dic.keys():
        print(dic[key][0])
    
def unpickle(file):
    """Load the pickle

    Args:
        file (str): path to the file to unpickle

    Returns:
        dict: the dictionary unpickled from the CIFAR10 or CIFAR100 directory
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def to_image(vector):
    """convert a vector loaded from CIFAR10 or CIFAR100 to an image with values between 0 and 1

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    colored_vector = vector.reshape(3, -1)
    image= colored_vector.reshape(3,32,32)
    image = np.transpose(image, (1, 2, 0))
    image = image/255
    return image
    
# Defining a function for shuffling
def common_shuffler(arr1, arr2):
    """This function will output a random tuple of elt from arr1 arr2 in the same place

    Args:
        arr1 (np.array)
        arr2 (np.array)

    Returns:
        tuple
    """
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]
    
def get_data_100():
    def take_down_not_used100(dataset):
        del dataset[b'coarse_labels']
        del dataset[b'filenames']
    path='cifar-100-python'
    dataset_test = unpickle(path+ '/test')
    dataset_train = unpickle(path+ '/train')
    take_down_not_used100(dataset_test)
    take_down_not_used100(dataset_train)
    labels_test = dataset_test.keys()
    labels_train = dataset_train.keys()
    names = unpickle(path+ '/meta')
    names = names[b'fine_label_names']
    # We need b'fine labels' and b'data' for our analysis
    x_train = dataset_train[b'data']
    x_test = dataset_test[b'data']
    y_train = np.array(dataset_train[b'fine_labels'])
    y_test = np.array(dataset_test[b'fine_labels'])
    
    return  x_train, x_test, y_train, y_test, names

def get_data_10():
    def take_down_not_used10(dataset):
        del dataset[b'batch_label']
        del dataset[b'filenames']
    path='cifar-10-batches-py'
    names = unpickle(path+ '/batches.meta')
    names = names[b'label_names']
    dataset_train={b'data' : [], b'labels' : []}
    for i in range(5):
        dataset_i = unpickle(path+ '/data_batch_' + str(i+1))
        for key in dataset_train.keys():
            dataset_train[key].extend(dataset_i[key])
    dataset_test = unpickle(path+ '/test_batch')
    take_down_not_used10(dataset_test)
    x_train = np.array(dataset_train[b'data'])
    x_test = dataset_test[b'data']
    y_train = np.array(dataset_train[b'labels'])
    y_test = np.array(dataset_test[b'labels'])
    
    return  x_train, x_test, y_train, y_test, names

def get_x_y(num):
    get_data =lambda x : None
    if num==10:
        get_data = get_data_10
    elif num==100:
        get_data = get_data_100
    else:
        raise ValueError('num for CIFAR should be 10 or 100')
    x_train, x_test, y_train, y_test, names = get_data()
    print('done unpickle')
    
    x_train = np.array([to_image(x_train[i]) for i in range(x_train.shape[0])], dtype= np.float32)
    x_test = np.array([to_image(x_test[i]) for i in range(x_test.shape[0])], dtype= np.float32)
    return x_train, x_test, y_train, y_test, names


def show_images():
    
    x_train, x_test, y_train, y_test, names = get_x_y(10)
    x_train_shuffle, y_train_shuffle = common_shuffler(x_train, y_train)
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train_shuffle[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(names[y_train_shuffle[i]])
    plt.show()
if __name__ == '__main__':
    show_images()