import numpy as np
from collections import OrderedDict
import config


PCBitMask = 0
for _ in range(config.nrFeatureBits):
    PCBitMask = (PCBitMask << 1) | 0x1

# constants
PC_BIT_MASK = PCBitMask
LOW_ORDER_SIX_BIT_MASK = 0x3F #0x3F = 0000 ... 0011 1111
BLOCK_INDEX_BITS = 6
BLOCK_OFFSET_BITS = 6
QUEUE_SIZE = config.lookahead + 1
QUEUE_COUNTER_OFFSET = QUEUE_SIZE * BLOCK_INDEX_BITS


class QueueSystem: # LRU policy
 
    # initialising capacity
    def __init__(self, capacity: int):
        self.queues = OrderedDict()
        self.capacity = capacity
        self.slotsField = 0
        for _ in range(QUEUE_SIZE): # from 0 to lookahead (i.e. lookahead+1 times)
            self.slotsField = (self.slotsField << 6) | LOW_ORDER_SIX_BIT_MASK
 
    # we return the value of the queueName
    # that is queried (in O(1)) and return -1 if we
    # don't find the queueName in out dict / queues.
    # We also move the queueName to the end
    # to show that it was recently used.
    def get(self, queueName: int) -> int:
        if queueName not in self.queues:
            return -1
        else:
            self.queues.move_to_end(queueName)
            return self.queues[queueName]
 
    # first, we add / update the queueName,
    # and also move the queueName to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first queueName (least recently used).
    def __put(self, queueName: int, value: int) -> None:
        self.queues[queueName] = value
        self.queues.move_to_end(queueName)
        if len(self.queues) > self.capacity:
            self.queues.popitem(last = False)

    def add(self, queueName: int, blockIndex: int) -> None:
        """ 
        Add block index to queue "queueName" 
        (creates the queue if it does not already exist)
        """
        queue = self.get(queueName)
        if queue == -1: # create new queue
            # place queue counter higher than the queue slots, and add block index
            queue = 1 << QUEUE_COUNTER_OFFSET | blockIndex
            self.__put(queueName, queue)
        else: # add to existing queue
            counterVal = self.getCounterVal(queue)
            # if queue is not already filled, increment counter
            if counterVal < QUEUE_SIZE:
                counterVal += 1
            # add block index to end of queue, and if there are more block indices
            # than queue slots, discard the block that was at the front of the queue
            uppdatedSlots = (queue << BLOCK_INDEX_BITS) & self.slotsField | blockIndex
            # append queue slots to queue counter
            queue = counterVal << QUEUE_COUNTER_OFFSET | uppdatedSlots
            self.__put(queueName, queue)

    def getCounterVal(self, queue:int) -> int:
        return queue >> QUEUE_COUNTER_OFFSET

    def end(self, queue:int) -> int:
        return queue & LOW_ORDER_SIX_BIT_MASK

    def front(self, queue:int) -> int:
        return (queue >> (BLOCK_INDEX_BITS * config.lookahead)) & LOW_ORDER_SIX_BIT_MASK



def bitfield(n,inputLength):
    # make sure that the binary representation is the same length as the input to the TCN
    binary = f'{n:0{inputLength}b}'
    # transform into bitarray
    bitarray = [1 if digit == '1' else 0 for digit in binary]
    return bitarray

def addBlockIdx(input, blockIndex) -> int:
    # make room for next block index | add the block index
    return (input << BLOCK_INDEX_BITS) | blockIndex

def getBlockIdx(input:int) -> int:
    return (input >> BLOCK_INDEX_BITS) & LOW_ORDER_SIX_BIT_MASK



def dataFormatorTrain(data):

    queues = QueueSystem(config.nrQueues)

    x_train = []
    y_train = []
    
    for (_, _, load_addr, load_ip, _) in data:

        pageAddr = load_addr >> (BLOCK_INDEX_BITS + BLOCK_OFFSET_BITS)

        blockIndex = getBlockIdx(load_addr)

        inputInfo = 0
        if config.feature == config.version.PAGE:
            queues.add(pageAddr, blockIndex)
            queue = queues.get(pageAddr)
            inputInfo = pageAddr
        else: # PC
            queues.add(load_ip, blockIndex)
            queue = queues.get(load_ip)
            # make sure PC is max "config.nrFeatureBits" bit
            inputInfo = load_ip & PC_BIT_MASK

        # check if queue is full
        if queues.getCounterVal(queue) >= QUEUE_SIZE:

            """
            Front of queue (oldest entry). 
            """
            blockIndexFront = queues.front(queue)
            # concatenate block index to the input information for the TCN
            inputInfo = addBlockIdx(inputInfo, blockIndexFront)
            # transform address into an array of its binary representation
            bitarray = bitfield(inputInfo, config.inputLength)
            # append TCN input
            x_train.append(bitarray)

            """
            End of queue (latest entry).
            """
            blockIndexEnd = queues.end(queue)
            # append label for the block in the active queue 
            # that was stored lookahead steps in the past.
            y_train.append(blockIndexEnd)

         
    #  Convert to np arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)


    return x_train, y_train


def dataFormatorGenerate(data):

    x_pred = []

    for (_, _, load_addr, load_ip, _) in data:

        inputInfo = 0
        if config.feature == config.version.PAGE:
            inputInfo = load_addr >> BLOCK_OFFSET_BITS
        else: # PC
            # make sure PC is max "config.nrFeatureBits" bit
            inputInfo = load_ip & PC_BIT_MASK
            # get block index from current address
            blockIndex = getBlockIdx(load_addr)
            # concatenate block index to the input information to the TCN
            inputInfo = addBlockIdx(inputInfo, blockIndex)

        # transform address into an array of its binary representation
        bitarray = bitfield(inputInfo, config.inputLength)
        # append TCN input
        x_pred.append(bitarray)         

    #  Convert to np array
    x_pred = np.array(x_pred)

    return x_pred