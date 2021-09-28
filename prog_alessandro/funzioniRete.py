from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import statistics as stat
import funzioni as fun
import funzioniIO as io

ciao = 0

def forwardPropagation(net, input):

    for i in range(net.numeroStrati):
        if i == 0: # primo strato, faccio il prodotto con l'input
            net.inputStrato[i] = np.dot(net.pesiStrato[i], np.transpose(input)) + net.bias[i]

        else: # altri strati, faccio il prodotto con l'output dello strato precedente
            net.inputStrato[i] = np.dot(net.pesiStrato[i], net.outputStrato[i - 1]) + net.bias[i]

        net.outputStrato[i] = fun.funzioniAttivazione[net.funzioneAttivazioneStrato[i]](net.inputStrato[i]) # applico la funzione di attivazione all'input

    return net, net.outputStrato[net.numeroStrati - 1]


def calcoloDerivate(net, target, input):

    net, output = forwardPropagation(net, input) # ottengo i valori di input e output per ogni nodo

    derivateBias = []
    derivatePesi = []
    delta = []

    # inizializzo tutti a zero
    for i in range(net.numeroStrati):
        delta.append(np.zeros((len(net.outputStrato[i]), len(net.outputStrato[i][0]))))
        derivatePesi.append(np.zeros((len(net.pesiStrato[i]), len(net.pesiStrato[i][0]))))
        derivateBias.append(np.zeros((len(net.bias[i]), 1)))

    # calcolo dei delta ( BACK PROPAGATION )
    for i in reversed(range(net.numeroStrati)):  # la back propagation parte dall'ultimo strato e sale al primo
        delta[i] = fun.derivateFunzioniAttivazione[net.funzioneAttivazioneStrato[i]](net.inputStrato[i])  # applico la derivata della funzione di attivazione dello strato corrente all'input di ogni neurone
        if i == (net.numeroStrati - 1):  # strato di output, moltiplico per la derivata della funzione di errore
            delta[i] = delta[i] * fun.derivateFunzioniErrore[net.funzioneErrore](net.outputStrato[i], target)
        else:  # strato interno, applico la regola ricorrente del delta
            delta[i] = delta[i] * np.dot(np.transpose(net.pesiStrato[i + 1]), delta[i + 1])

    # calcolo delle derivate a partire dai delta
    sum = 0
    for i in range(net.numeroStrati):
        if i == 0: # primo strato
            derivatePesi[i] = (np.dot(delta[i], (input)))
        else: # altri strati
            derivatePesi[i] = (np.dot(delta[i], np.transpose(net.outputStrato[i - 1])))

        for nodo in range(net.numeroNodiStrato[i]):  # per i bias la derivata è proprio uguale a delta_i perchè il peso è pari ad 1
            for j in delta[i][nodo]:
                sum = sum + j
            derivateBias[i][nodo] = sum
            sum = 0

    return derivateBias, derivatePesi


def learningBatch(net, training, targetTraining, validation, targetValidation, eta, mu, numeroEpoche):

    bestErrore = np.inf
    bestNet = net
    errore = np.zeros(numeroEpoche)

    target = np.transpose(targetTraining)
    targetValidation = np.transpose(targetValidation)

    # variabili per salvare la variazione di pesi tra le epoche, utili per implementare il momento
    variazionePesiEpocaPrecedente = []
    variazionePesiEpocaCorrente = []
    variazioneBiasEpocaCorrente = []
    variazioneBiasEpocaPrecedente = []

    for i in range(net.numeroStrati): # le inizializzo con zero
        variazionePesiEpocaCorrente.append(np.zeros((len(net.pesiStrato[i]), len(net.pesiStrato[i][0]))))
        variazionePesiEpocaPrecedente.append(np.zeros((len(net.pesiStrato[i]), len(net.pesiStrato[i][0]))))
        variazioneBiasEpocaCorrente.append(np.zeros((net.numeroNodiStrato[i], 1)))
        variazioneBiasEpocaPrecedente.append(np.zeros((net.numeroNodiStrato[i], 1)))

    for epoca in range(numeroEpoche):

        derivateBias, derivatePesi = calcoloDerivate(net, target, training)

        for i in range(net.numeroStrati):

            # calcolo della variazione di peso tramite la formula del momento
            variazionePesiEpocaCorrente[i] = (- eta * derivatePesi[i]) + (mu * variazionePesiEpocaPrecedente[i])
            variazioneBiasEpocaCorrente[i] = (- eta * derivateBias[i]) + (mu * variazioneBiasEpocaPrecedente[i])

            # aggiornamento dei pesi
            net.pesiStrato[i] = net.pesiStrato[i] - eta + variazionePesiEpocaCorrente[i]
            net.bias[i] = net.bias[i] - eta + variazioneBiasEpocaCorrente[i]

            variazionePesiEpocaPrecedente[i] = variazionePesiEpocaCorrente[i]
            variazioneBiasEpocaPrecedente[i] = variazioneBiasEpocaCorrente[i]

        net, output = forwardPropagation(net, validation)

        errore[epoca] = fun.funzioniErrore[net.funzioneErrore](output, targetValidation) # calcolo l'errore sul validation set
        print(errore[epoca])

        if errore[epoca] < bestErrore: # mi salvo la rete con l'errore minimo
            bestErrore = errore[epoca]
            bestNet = deepcopy(net)

        #print("ERRORE EPOCA " + str(epoca) + " : " + str(errore[epoca]))

    return bestNet


def testaRete(net, testSet, valoreAtteso, descrizione):

    net, output = forwardPropagation(net, testSet) # calcolo i valori di output usando il test set

    output = np.transpose(output)

    risultato = [] # array contente la mia classificazione, prendo il neurone di output con valore massimo

    for i in range(len(output)):
        risultato.append(list(output[i]).index(np.amax(output[i])))

    # valuto la mia classificazione
    accuracy = accuracy_score(risultato, valoreAtteso)
    precision = precision_score(risultato, valoreAtteso, average='macro', zero_division=0)
    recall = recall_score(risultato, valoreAtteso, average='macro', zero_division=0)
    f1 = f1_score(risultato, valoreAtteso, average='macro', zero_division=0)

    corretti = accuracy_score(risultato, valoreAtteso, normalize=False)
    totali = len(risultato)

    io.printScore(accuracy, precision, recall, f1, descrizione, corretti, totali)

    scores = [accuracy, precision, recall, f1]

    return scores


def calcolaMedia(scores):

    scores = np.array(scores)
    scores = np.transpose(scores)

    scoresMean = []
    stdDeviation = []

    for i in range(len(scores)):
        scoresMean.append(stat.mean(scores[i]))
        stdDeviation.append(stat.stdev(scores[i]))

    io.printScore(scoresMean[0], scoresMean[1], scoresMean[2], scoresMean[3], "Media delle valutazioni  |  ")
    io.printStdDeviation(stdDeviation)

    return scoresMean
