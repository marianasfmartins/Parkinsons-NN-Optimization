import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.neural_network import MLPClassifier
import numpy as np
from project_data import X_train, Y_train, X_val, Y_val

#training the data and getting the predictions
def get_predictions(solution, X_train, Y_train, X_val, Y_val):
    
    NN = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                       alpha=0.0001, learning_rate='constant', max_iter=1)
    #training our NN
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


    



