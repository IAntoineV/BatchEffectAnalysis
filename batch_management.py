import math
import numpy as np
from random import shuffle

import tensorflow as tf


def split_data_by_class(x_set : np.array, y_set : np.array, list_of_class : list[int]):
    """This function is used to split the data by keeping only the classes in list_of_class.
    Each class has its own data base in lis_per_class in the output.

    Args:
        x_set (np.array): _description_
        y_set (np.array): _description_
        list_of_class (list[int]): _description_

    Returns:
        _type_: list[list[np.array]]
        
        the output contain a list (for each class in list_of_class) of list of samples.
        It is a splitted version of x_set.
    """
    classes = sorted(list_of_class)
    nb_classes = len(classes)
    # map classes index from input to [0, nb_classes-1].
    mapping = {classes[k] : k for k in range(nb_classes)}
    nb_samples = x_set.shape[0]
    # Initialize the splitted data.
    list_per_class = [[] for i in range(nb_classes)]
    # We extract the data from x_set to split the dataset into classes sub datasets.
    for i in range(nb_samples):
        if y_set[i] in mapping.keys():
            index_store=mapping[y_set[i]]
            list_per_class[index_store].append(x_set[i])
    return list_per_class
class CIFARSequence_shuffle(tf.keras.utils.Sequence):
    """Class is used to generate the batch used by the fit.
    
        it generate random batch of data

        Args:
            x_set (np.array): training data
            y_set (np.array): training labels
            list_of_class (list[int]): class considered
            max_nb_class (int): number of classes in our training data           .
        """
    def __init__(self, x_set, y_set, list_of_class, max_nb_class):
        # If no list_of_class is provided, we use all classes.
        if list_of_class is None:
            list_of_class = [k for k in range(max_nb_class)]
        # The number of classes considered will be the batch size.
        self.nb_classes  = len(list_of_class)
        # We get the splitted data
        list_per_class = split_data_by_class(x_set, y_set, list_of_class)
        L = []
        # We rejoin data with there label to shuffle the data.
        for k,data in enumerate(list_per_class):
            for i in range(len(data)):
                L.append([data[i],k])
        shuffle(L)
        self.L = L
        x, y = zip(*L)
        x, y = list(x), list(y)
        self.x = np.array(x)
        self.y = np.array(y)
        self.nb_samples = self.x.shape[0]
        

    def __len__(self):
        """Mandatory attribute for tf.keras.utils.Sequence class. 

        Returns:
            nb_of_step : The number of step of each epoch in the .fit
        """
        return math.ceil(len(self.x) / self.nb_classes)

    def __getitem__(self, idx):
        """Mandatory attribute for tf.keras.utils.Sequence class. 

        Returns:
            batch_x, batch_y : (numpy.array, numpy.array)
            the batch of x and y to compute the loss and gradient from.
        """
        low = idx * self.nb_classes
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.nb_classes, self.nb_samples)
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        return batch_x, batch_y
    def on_epoch_end(self):
        shuffle(self.L)
        x, y = zip(*self.L)
        x, y = list(x), list(y)
        self.x = np.array(x)
        self.y = np.array(y)
        
def same_len(L : list[list]):
    """We check if all elements of the list have the same length.

    Args:
        L (list of list)

    Returns:
        None
    """
    if len(L) == 0:
        return True
    length=len(L[0])
    boolean= True
    for elt in L:
        boolean = boolean and len(elt) == length
    if not boolean:
        raise ValueError("All elements of the list must have the same length for sequence_uniform") 

class CIFARSequence_uniform(tf.keras.utils.Sequence):
    """Class is used to generate the batch used by the fit.
    
        it generate batch with each class appearing once in each batch

        Args:
            x_set (np.array): training data
            y_set (np.array): training labels
            list_of_class (list[int]): class considered
            max_nb_class (int): number of classes in our training data           .
        """
    def __init__(self, x_set, y_set, list_of_class, max_nb_class):
        if list_of_class is None:
            list_of_class = [k for k in range(max_nb_class)]
        list_per_class = split_data_by_class(x_set, y_set, list_of_class)
        
        classes = sorted(list_of_class)
        self.classes=classes
        self.nb_classes = len(classes)
        for k in range(self.nb_classes):
            shuffle(list_per_class[k])
        same_len(list_per_class)
        self.list_per_class = list_per_class
        self.nb_elt = len(list_per_class[0])
        stack = np.array(list_per_class)
        self.dissociated_x = stack

    def __len__(self):
        """Mandatory attribute for tf.keras.utils.Sequence class. 

        Returns:
            nb_of_step : The number of step of each epoch in the .fit
        """
        return self.nb_elt

    def __getitem__(self, idx):
        """Mandatory attribute for tf.keras.utils.Sequence class. 

        Returns:
            batch_x, batch_y : (numpy.array, numpy.array)
            the batch of x and y to compute the loss and gradient from.
        """
        batch_x = np.array(self.dissociated_x[:, idx])
        batch_y = np.array([self.classes[k] for k in range(self.nb_classes)])

        return batch_x, batch_y
    def on_epoch_end(self):
        for i in range(self.nb_classes):
            shuffle(self.list_per_class[i])
        self.dissociated_x = np.array(self.list_per_class)


def focus_mode(idx,nb_batch_per_class, nb_classes):
    """ Mode used in the CIFARSequence_focus class.
    In this mode, the index are defined to alternate between batch of each classes. (batch of only class 1 samples, 
    then batch of only class 2 samples, etc., up until batch of only class nb_classes-1 samples, and we loop...)

    Args:
        idx (_type_): the index of step of the epoch during fit.
        nb_batch_per_class (_type_): number of batch of each classes
        nb_classes (_type_): number of classes

    Returns:
        i,j : tuple(int,int)
        i represent the index of the class and j represent the index of the class's batch.
        
    """
    return idx % nb_classes, idx // nb_classes
def fullfocus_mode(idx,nb_batch_per_class, nb_classes):
    """ Mode used in the CIFARSequence_focus class.
    In this mode, the index are defined to git on all batch of one class then go to the next class. 
    (batch of only class 1 samples, then batch of only class 1 samples, etc., up until there is no more batch of class 1.
    Then, we check for all batches of class 2, etc...)

    Args:
        idx (_type_): the index of step of the epoch during fit.
        nb_batch_per_class (_type_): number of batch of each classes
        nb_classes (_type_): number of classes

    Returns:
        i,j : tuple(int,int)
        i represent the index of the class and j represent the index of the class's batch.
        
    """
    return idx // nb_batch_per_class, idx % nb_batch_per_class
class CIFARSequence_focus(tf.keras.utils.Sequence):
    """Class is used to generate the batch used by the fit.
    
        it generate batch with only one class in each batch.

        Args:
            x_set (np.array): training data
            y_set (np.array): training labels
            list_of_class (list[int]): class considered
            max_nb_class (int): number of classes in our training data  
            batch_size (int): batch size, it needs to divide the number of samples in each class.
            mode (function): function to generate in which order we consider the batches. Implemented :
            focus_mode
            fullfocus_mode
        """
    def __init__(self, x_set, y_set, list_of_class, max_nb_class, batch_size,mode):
        self.mode = mode
        self.batch_size = batch_size
        if list_of_class is None:
            list_of_class = [k for k in range(max_nb_class)]
        list_per_class = split_data_by_class(x_set, y_set, list_of_class)
        
        classes = sorted(list_of_class)
        self.classes=classes
        self.nb_classes = len(classes)
        for k in range(self.nb_classes):
            shuffle(list_per_class[k])
        same_len(list_per_class)
        self.nb_elt = len(list_per_class[0])
        stack = np.array(list_per_class)
        self.dissociated_x = stack
        self.nb_batch_per_class = math.ceil(self.nb_elt / self.batch_size)
        print('nb step', self.nb_classes * self.nb_batch_per_class, self.nb_classes, self.nb_batch_per_class  )
    def __len__(self):
        """Mandatory attribute for tf.keras.utils.Sequence class. 

        Returns:
            nb_of_step : The number of step of each epoch in the .fit
        """
        return self.nb_classes * self.nb_batch_per_class

    def __getitem__(self, idx):
        """Mandatory attribute for tf.keras.utils.Sequence class. 

        Returns:
            batch_x, batch_y : (numpy.array, numpy.array)
            the batch of x and y to compute the loss and gradient from.
        """
        i, j = self.mode(idx, self.nb_batch_per_class, self.nb_classes)
        low = j *self.batch_size
        high = min(low + self.batch_size,self.nb_elt)
        batch_x = np.array(self.dissociated_x[i, low:high] )
        batch_y = np.array([self.classes[i] for k in range(high - low)])

        return batch_x, batch_y