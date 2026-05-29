from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from project_data import data
import numpy as np


#data partition
def data_partition(data,target_name):
    X=data.drop(columns=target_name)
    Y=data[target_name]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                      Y_train,
                                                      test_size=0.25, random_state=15, shuffle=True, stratify=Y_train
                                                      )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
#Data Partitioning
X_train, X_val, X_test, Y_train, Y_val, Y_test=data_partition(data,'status')

#training the data and getting the predictions
def get_predictions(solution, X_train, Y_train, X_val, Y_val):
    # Passo 1: Inicializar a rede e treinar 1 iteração para criar a estrutura interna
    NN = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                       alpha=0.0001, learning_rate='constant', max_iter=1)
    NN.fit(X_train, Y_train)

    # Passo 2: Partir o vetor linear do lobo nas matrizes de pesos (coefs_)
    idx = 0
    new_coefs = []
    for coef in NN.coefs_:
        size = coef.size
        new_coefs.append(np.array(solution[idx:idx + size]).reshape(coef.shape))
        idx += size

    # Passo 3: Partir o vetor linear do lobo nos vetores de bias (intercepts_)
    new_intercepts = []
    for intercept in NN.intercepts_:
        size = intercept.size
        new_intercepts.append(np.array(solution[idx:idx + size]).reshape(intercept.shape))
        idx += size

    # Passo 4: Injetar os pesos do lobo na rede neural
    NN.coefs_ = new_coefs
    NN.intercepts_ = new_intercepts

    # Passo 5: Fazer previsões com os pesos do lobo (sem treinar novamente)
    Y_pred = NN.predict(X_val)
    return Y_pred


def get_n_weights(X_train, Y_train=None):
    """Calcula o número total de pesos usando a fórmula definida pelo grupo:
    (d × 100) + 100 + (100 × 2) + 2  =  100d + 302
    Onde d = número de features de entrada, 100 = neurónios ocultos, 2 = classes de saída.
    """
    d = X_train.shape[1]  # número de features de entrada
    return (d * 100) + 100 + (100 * 2) + 2



