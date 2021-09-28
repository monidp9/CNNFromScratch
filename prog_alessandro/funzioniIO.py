from pip._vendor.distlib.compat import raw_input
import sys


def printConfigurazioneRete (numeroStrati, numeroNodiStrato, funzioneStrato, funzioneErrore, numeroEpoche, dimensioneTraining, dimensioneTest, dimensioneValidation, numeroFolds, eta, mu):

    print("")

    print("Numero strati: " + str(numeroStrati))
    for i in range(numeroStrati):
        print("Strato " + str(i+1))
        print("\tNumero nodi: " + str(numeroNodiStrato[i]))
        print("\tFunzione di attivazione: " + getNomeFunzioneAttivazione(funzioneStrato[i]))
    print("Funzione di errore: " + getNomeFunzioneErrore(funzioneErrore))
    print("Numero epoche: " + str(numeroEpoche))
    print("Dimensione training set: " + str(dimensioneTraining))
    print("Dimensione test set: " + str(dimensioneTest))
    print("Dimensione validation set: " + str(dimensioneValidation))
    print("Numero di folds: " + str(numeroFolds))
    print("Eta: " + str(eta))
    print("Mu: " + str(mu))


def getInput (inputName, defaultValue, minValue, maxValue = sys.maxsize):

    while True:
        try:
            returnValue = int(raw_input(inputName) or defaultValue)
            if not minValue <= returnValue <= maxValue:
                raise ValueError
        except ValueError:
            print('ERRORE: Input non valido!')
        else:
            break
    return returnValue


def getNomeFunzioneAttivazione (num):
    if num == 0:
        return "Identità"
    elif num == 1:
        return "Sigmoide"
    else:
        return "ReLU"


def getNomeFunzioneErrore (num):
    if num == 0:
        return "Somma dei quadrati"
    else:
        return "Cross Entropy Soft Max"


def printScore(accuracy, precision, recall, f1, descrizione = "",  corretti = -1, totali = -1):

    format  = "{:1.3f}"

    accuracy = format.format(accuracy)
    precision = format.format(precision)
    recall = format.format(recall)
    f1 = format.format(f1)

    if totali == -1:
        accuracyDeatiled = ""
    else:
        accuracyDeatiled = " (" + str(corretti) + " / " + str(totali) + ")"

    print(descrizione + "Accuratezza: " + accuracy + accuracyDeatiled + "  Precision: " + precision + "  Recall: " + recall + "  F1 Score: " + f1)


def printStdDeviation(stdDeviation):

    format  = "{:1.3f}"
    accuracy = format.format(stdDeviation[0])
    precision = format.format(stdDeviation[1])
    recall = format.format(stdDeviation[2])
    f1 = format.format(stdDeviation[3])

    print("Deviazione standard      |  Accuratezza: " + accuracy + "  Precision: " + precision + "  Recall: " + recall + "  F1 Score: " + f1)
