from mnist import MNIST
from sklearn.model_selection import KFold
import numpy as np
import Rete as rete
import funzioniRete as fr
import funzioniIO as io

np.seterr(all='ignore')

# parametri della discesa del gradiente con momento
eta = 0.0005
mu = 0.1


useDefault  = io.getInput("Usare valori predefiniti? (default: Si)\n\t1) Si\n\t2) No\n\tScelta: ", 1, 1, 2)
if useDefault == 2: useDefault = 0

if useDefault:

    useMNIST = 1
    numeroStrati = 2
    numeroNodiStrato = np.ndarray(int(numeroStrati), dtype=int)
    funzioneStrato = np.ndarray(int(numeroStrati), dtype=int)
    numeroNodiStrato[0] = 100
    numeroNodiStrato[1] = 10
    funzioneStrato[0] = 1
    funzioneStrato[1] = 0
    funzioneErrore = 1
    numeroEpoche = 50
    dimensioneTest = 4000
    dimensioneTraining = 4500
    dimensioneValidation = 500
    numeroFolds = 10

else:

    numeroStrati = io.getInput("Numero strati (default: 2): ", 2, 1)

    numeroNodiStrato = np.ndarray(int(numeroStrati), dtype=int)
    funzioneStrato = np.ndarray(int(numeroStrati), dtype=int)

    for i in range(int(numeroStrati)):
        numeroNodiStrato[i] = io.getInput("Numero nodi " + str(i + 1) + "° strato (default: 10): ", 10, 1)
        funzioneStrato[i] = io.getInput("Funzione di attivazione " + str(i + 1) + "° strato (default: Sigmoide)\n\t1) Identità\n\t2) Sigmoide\n\t3) ReLU\n\tScelta: ", 2, 1, 3) - 1

    funzioneErrore = io.getInput("Funzione di errore (default: Cross Entropy Soft Max)\n\t1) Somma dei quadrati\n\t2) Cross Entropy Soft Max\n\tScelta: ", 2, 1, 2) - 1

    numeroEpoche = io.getInput("Numero Epoche (default: 50): ", 50, 1)

    dimensioneTest = io.getInput("Dimensione del Test Set (default: 4000): ", 4000, 1)

    dimensioneTraining = io.getInput("Dimensione del Training Set (default: 5000): ", 5000, 1)

    dimensioneValidation = io.getInput("Dimensione del Validation Set (default: 500): ", 1, 1)

    numeroFolds = io.getInput("Numero di folds (default: 10): ", 10, 2)


# path di dove si trovano i file MNIST
# caricamento dataset
mnistData = MNIST('./python-mnist/data')

# caricamento del training set
trainingMNIST, trainingAttesoMNIST = mnistData.load_training()
testMNIST, testAttesoMNIST = mnistData.load_testing()

# creo un array di numeri compresi tra 1 e la lunghezza del set
randnumsTraining = np.random.randint(0, len(trainingMNIST), dimensioneTraining)
randnumsValidation = np.random.randint(0, len(trainingMNIST), dimensioneValidation)
randnumsTest = np.random.randint(0, len(testMNIST), dimensioneTest)

# creo training, validation, test set e li riduco
trainingSet = np.array(trainingMNIST)
trainingSet = trainingSet[randnumsTraining]
valoreAttesoTraining = np.array(trainingAttesoMNIST)
valoreAttesoTraining = valoreAttesoTraining[randnumsTraining]

validationSet = np.array(trainingMNIST)
validationSet = validationSet[randnumsValidation]
valoreAttesoValidation = np.array(trainingAttesoMNIST)
valoreAttesoValidation = valoreAttesoValidation[randnumsValidation]

testSet = np.array(testMNIST)
testSet = testSet[randnumsTest]
valoreAttesoTest = np.array(testAttesoMNIST)
valoreAttesoTest = valoreAttesoTest[randnumsTest]

# creo il target del training e validation della rete basandomi sul valore atteso. Metto 1 se la classe è corretta 0 altrimenti
targetTraining = np.zeros((dimensioneTraining, numeroNodiStrato[numeroStrati - 1]))
for i in range(dimensioneTraining):
    targetTraining[i][valoreAttesoTraining[i]] = 1

targetValidation = np.zeros((dimensioneValidation, numeroNodiStrato[numeroStrati - 1]))
for i in range(dimensioneValidation):
    targetValidation[i][valoreAttesoValidation[i]] = 1

# dimensioni prese dal dataset MNIST
dimensioneFeatures = len(trainingSet[0])


# creo la rete
miaRete = rete.net(dimensioneTraining, dimensioneFeatures, numeroStrati, numeroNodiStrato, funzioneStrato, funzioneErrore)
# print(miaRete.pesiStrato[1])

# stampa configurazione rete
io.printConfigurazioneRete(numeroStrati, numeroNodiStrato, funzioneStrato, funzioneErrore, numeroEpoche, dimensioneTraining, dimensioneTest, dimensioneValidation, numeroFolds, eta, mu)


print("\nLearning con K Fold...")

kf = KFold(n_splits=numeroFolds)
k = 1
scores = []

for trainIndex, testIndex in kf.split(trainingSet):

    miaRete = fr.learningBatch(miaRete, trainingSet[trainIndex], targetTraining[trainIndex], validationSet, targetValidation, eta, mu, numeroEpoche)

    scores.append(fr.testaRete(miaRete, trainingSet[testIndex], valoreAttesoTraining[testIndex], "Fold " + str(k).zfill(2) + "  |  "))
    k = k + 1


print("")
fr.calcolaMedia(scores)
