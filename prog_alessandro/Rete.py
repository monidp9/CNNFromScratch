import numpy as np

class net:

    def __init__(self, dimensioneInput, numeroFeatures, numeroStrati, numeroNodiStrato, funzioni, funzioneErrore):

        self.dimensioneInput = dimensioneInput
        self.numeroFeatures = numeroFeatures
        self.numeroStrati = numeroStrati
        self.numeroNodiStrato = numeroNodiStrato

        self.pesiStrato = []
        for i in range(numeroStrati):
            if i == 0:
                self.pesiStrato.append(np.random.normal(size=(numeroNodiStrato[i], self.numeroFeatures)))
            else:
                self.pesiStrato.append(np.random.normal(size=(numeroNodiStrato[i], numeroNodiStrato[i - 1])))

        self.funzioneAttivazioneStrato = []
        for i in range(numeroStrati):
            self.funzioneAttivazioneStrato.append(funzioni[i])

        self.bias = []
        for i in range(numeroStrati):
            self.bias.append(np.random.normal(size=(numeroNodiStrato[i], 1)))

        self.inputStrato = []
        self.outputStrato = []

        for i in range(int(numeroStrati)):
            self.inputStrato.append(np.zeros((numeroNodiStrato[i], self.dimensioneInput)))
            self.outputStrato.append(np.zeros((numeroNodiStrato[i], self.dimensioneInput)))

        self.funzioneErrore = funzioneErrore