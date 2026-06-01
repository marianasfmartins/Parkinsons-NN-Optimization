#TESTING STUFF

from GA_operators.crossover import *
from GA_operators.selector import *
from GA_operators.mutators import *
from GA_operators.selector import *
from GA_operators.population import *
from utils import *
from sklearn.neural_network import MLPClassifier
import numpy as np
from project_data import X_train, Y_train, X_val, Y_val
from utils import fitness_misclassification


mlp = MLPClassifier(hidden_layer_sizes=(100,))

# 2. Faz um fit rápido com dados fictícios (X_train deve ter o mesmo número de colunas do teu dataset)
# Apenas para o sklearn gerar as matrizes 'coefs_' e 'intercepts_' com os shapes corretos
mlp.fit(X_train[:2], Y_train[:2]) 

# 3. Calcula o tamanho total que o teu vetor de pesos deve ter
num_pesos = sum(w.size for w in mlp.coefs_)
num_biases = sum(b.size for b in mlp.intercepts_)
tamanho_total_do_vetor = num_pesos + num_biases

print(f"O teu vetor deve ter exatamente {tamanho_total_do_vetor} elementos.")

population=initialize_population(pop_size=30, n_weights=tamanho_total_do_vetor, method='he_uniform', n_in=X_train.shape[0], n_out=Y_train.shape[0])
 
fitness=fitness_misclassification(population[2], X_train, Y_train, X_val, Y_val)
print(fitness)





