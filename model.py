from abc import abstractmethod
from tcn import compiled_tcn
from tensorflow.keras.utils import Sequence
from tensorflow import keras

import numpy as np
import math
import config
from dataFormator import dataFormatorTrain
from dataFormator import dataFormatorGenerate

 

class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass


class FixedOffset(MLPrefetchModel):

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Has no model to load')

    def save(self, path):
        # Save your model to a file
        print('Has no model to save')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for Fixed-Offset')
        prefetches = []
        for (instr_id, _, load_addr, _, _) in data:
            # Prefetch the next two blocks
            #prefetches.append((instr_id, ((load_addr >> 6) + 2) << 6))
            prefetches.append((instr_id, ((load_addr >> 6) + 3) << 6))
            
        return prefetches


class DataGenerator(Sequence):
    """
    Generates data for Keras Sequence based data generator. 
    Suitable for building data generator for training and prediction.
    """

    def __init__(self, x,  y=0, batch_size=32, to_fit=True):
        """
        Initialization 
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        """
        self.x = x
        self.y = y
        self.to_fit = to_fit
        self.batch_size = batch_size

    def __len__(self):
        """
        Denotes the number of batches per epoch 
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        indexes = np.arange(len(self.x))
        # Generate indexes of the batch
        indexes = indexes[index *
                          self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        X = self.x[indexes]

        if self.to_fit:
            y = self.y[indexes]
            return X, y
        else:
            return X



# nr of dilations
nrDilations = math.ceil(math.log2((config.inputLength - 1)/(2 * (config.kernelSize - 1)) + 1))

receptiveField = 1+2*(config.kernelSize-1)*((2**nrDilations)-1)


def printConfig():
    print("\n-----------TCN Config-----------\n")
    print("Feature = ", config.feature)
    print("Receptive field = ", receptiveField)
    print("Input lenght = ", config.inputLength)
    print("Number of dilations = ", nrDilations)
    print("lookahead distance = ", config.lookahead)
    print("Kernel size = ", config.kernelSize)
    print("Number of filters = ", config.nrFilters)
    print("Number of queues = ", config.nrQueues)
    print("Number of residual = ", config.resBlocks)
    print("degree = ", config.degree)
    print("\n")


# class TCN(tf.Module):

#     def __init__(self):
#         super().__init__()
#         # Initialize TCN
#         print("Initializing TCN")
#         self.tcn = compiled_tcn(return_sequences=False,
#                                 num_feat=1,
#                                 num_classes=config.outputClasses,
#                                 nb_filters=config.nrFilters,
#                                 kernel_size=config.kernelSize,
#                                 dilations=[2 ** i for i in range(nrDilations)],
#                                 nb_stacks=config.resBlocks,
#                                 max_len=config.inputLength,
#                                 use_weight_norm=True,
#                                 use_skip_connections=True)
#         self.tcn.summary()

def TCN():
    print("Initializing TCN")
    tcn = compiled_tcn(return_sequences=False,
                        num_feat=1,
                        num_classes=config.outputClasses,
                        nb_filters=config.nrFilters,
                        kernel_size=config.kernelSize,
                        dilations=[2 ** i for i in range(nrDilations)],
                        nb_stacks=config.resBlocks,
                        max_len=config.inputLength,
                        use_weight_norm=True,
                        use_skip_connections=True)
    tcn.summary()
    return tcn


class TCNPrefetcher(MLPrefetchModel):
    """
    TCN prefetcher
    """
    def __init__(self):
        self.model = TCN()

    def load(self, path):
        self.model = keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

    def train(self, train_data):
        printConfig()
        # format for TCN
        x_train, y_train = dataFormatorTrain(train_data)
        # train the model
        self.model.fit(x_train, y_train, epochs=1, batch_size=config.trainBatchSize)

    def generate(self, pred_data):
        print("Generating for TCN")
        # format for TCN predictions
        x_pred = dataFormatorGenerate(pred_data)
       
        # create data generator
        print("Time for predictions")
        pred_generator = DataGenerator(x_pred, batch_size=config.predBatchSize, to_fit=False)

        # predicted block index probabilities
        predProb = []
        predProb = self.model.predict(pred_generator)

        # sort probabilities from lowest to highest
        sortedPredIndices = np.argsort(predProb)

        prefetchBlocks = []
        # for all prefetch predictions, get the "degree" number 
        # of labels with highest probabilities
        prefetchBlocks = sortedPredIndices[:,-config.degree:]

        prefetches = []
        for i in range(len(prefetchBlocks)):
            for j in range(config.degree):
                # clear page offset (12 low-order bits)
                # and insert the Block index into the Block index part of address
                predictedAddr = (pred_data[i][2] & ~0xFFF) | (prefetchBlocks[i][j] << 6)
                # issue prefetch      instr_id
                prefetches.append((pred_data[i][0], predictedAddr))

        return prefetches


# FixedOffset, TCNPrefetcher

Model = TCNPrefetcher
