import numpy as np
from scipy.special import softmax


def identita(x):
    return x


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x,0)


def derivataIdentita(x):
    len1 = len(x)
    len2 = len(x[0])
    return np.ones((len1, len2))


def derivataSigmoide(x):
    x = sigmoide(x)
    return x * (1 - x)


def derivataRelu(x):
    ret = np.copy(x)
    ret[ret >= 0] = 1
    ret[ret < 0] = 0
    return ret


def derivataSommaDeiQuadrati(output, atteso):
    return output - atteso


def derivataCrossEntropySoftMax(output, atteso):
    return softmax(output, axis=0) - atteso


def sommaDeiQuadrati(output, atteso):
    return 0.5 * np.sum(np.power(output - atteso, 2))


def crossEntropySoftMax(output, atteso):
    output = softmax(output, axis=0)
    return - np.sum(atteso * np.log(output))


funzioniAttivazione = [identita, sigmoide, relu]

derivateFunzioniAttivazione = [derivataIdentita, derivataSigmoide, derivataRelu]

funzioniErrore = [sommaDeiQuadrati, crossEntropySoftMax]

derivateFunzioniErrore = [derivataSommaDeiQuadrati, derivataCrossEntropySoftMax]
