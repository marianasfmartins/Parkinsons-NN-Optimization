import numpy as np
import sys
import os
import random

"""
Métodos de inicialização de pesos para a rede neural.

He / Kaiming (para ReLU):
  A função ReLU anula todos os valores negativos, o que reduz efetivamente
  a variância do sinal à metade a cada camada. O método de He compensa isso
  escalando os pesos com um fator de 2x — garantindo que o sinal não
  desaparece nem explode à medida que atravessa as camadas da rede.

  - He Normal  (Kaiming Normal):  pesos ~ N(0, sqrt(2 / n_in))
  - He Uniform (Kaiming Uniform): pesos ~ Uniform(-sqrt(6 / n_in), +sqrt(6 / n_in))

Xavier / Glorot (para tanh / sigmoid):
  Projetado para manter a variância do sinal estável tanto no forward pass
  como no backward pass. Usa fan_in + fan_out para equilibrar ambos os lados.

  - Xavier Normal:  pesos ~ N(0, sqrt(2 / (n_in + n_out)))
  - Xavier Uniform: pesos ~ Uniform(-sqrt(6 / (n_in + n_out)), +sqrt(6 / (n_in + n_out)))

Parâmetro n_in:
  Representa o número de entradas de cada neurónio (fan_in).
  Para o nosso problema: n_in = número de features = 21.

Parâmetro n_out:
  Representa o número de saídas de cada neurónio (fan_out).
  Para o nosso problema: n_out = 1 (saída binária).
"""

import numpy as np

def generate_random_solution(n_weights):  #generating our random solution as a start point for algorithms
    return[random.uniform(-0.05,0.05) for _ in range(n_weights)]


def he_normal_ind(n_weights, n_in=21):
    """
    Inicialização He Normal (Kaiming Normal).

    Cada peso é amostrado de uma distribuição normal com:
      média = 0
      desvio padrão = sqrt(2 / n_in)

    Args:
        n_weights (int): Número total de pesos a inicializar.
        n_in (int): Número de entradas do neurónio (fan_in). Default = 21 (features do dataset).

    Returns:
        numpy array de forma (n_weights,) com os pesos inicializados.
    """
    std = np.sqrt(2.0 / n_in)
    return np.random.normal(loc=0.0, scale=std, size=n_weights)


def he_uniform_ind(n_weights, n_in=21):
    """
    Inicialização He Uniform (Kaiming Uniform).

    Cada peso é amostrado de uma distribuição uniforme com:
      limite = sqrt(6 / n_in)
      intervalo = [-limite, +limite]

    Args:
        n_weights (int): Número total de pesos a inicializar.
        n_in (int): Número de entradas do neurónio (fan_in). Default = 21 (features do dataset).

    Returns:
        numpy array de forma (n_weights,) com os pesos inicializados.
    """
    limit = np.sqrt(6.0 / n_in)
    return np.random.uniform(low=-limit, high=limit, size=n_weights)


def xavier_normal_ind(n_weights, n_in=21, n_out=1):
    """
    Inicialização Xavier Normal (Glorot Normal).

    Cada peso é amostrado de uma distribuição normal com:
      média = 0
      desvio padrão = sqrt(2 / (n_in + n_out))

    Projetado para manter a variância do sinal estável em ambos os sentidos
    (forward e backward pass). Recomendado para ativações tanh e sigmoid.

    Args:
        n_weights (int): Número total de pesos a inicializar.
        n_in  (int): Fan-in  — número de entradas do neurónio. Default = 21.
        n_out (int): Fan-out — número de saídas do neurónio.  Default = 1.

    Returns:
        numpy array de forma (n_weights,) com os pesos inicializados.
    """
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(loc=0.0, scale=std, size=n_weights)



def initialize_population(pop_size, n_weights=2401, method='random', n_in=21, n_out=1):
    """
    Inicializa a população do GA.

    Cada indivíduo é um vetor numpy de n_weights pesos.
    O método de inicialização define como esses pesos são gerados —
    uma boa inicialização pode acelerar a convergência do algoritmo.

    Args:
        pop_size (int):  Número de indivíduos na população.
        n_weights (int): Número de pesos por indivíduo (dimensão do problema).
        method (str):    Método de inicialização:
                           'random'         — uniforme em [-0.05, 0.05]
                           'he_normal'      — He Normal  (para ReLU)
                           'he_uniform'     — He Uniform (para ReLU)
                           'xavier_normal'  — Xavier Normal  (para tanh/sigmoid)
        n_in (int):      Fan-in (nº de features de entrada). Usado pelo He e Xavier. Default=21.
        n_out (int):     Fan-out (nº de saídas). Usado pelo Xavier. Default=1.

    Returns:
        population (list): Lista de arrays numpy, cada um com n_weights valores.
    """
    population = []

    for _ in range(pop_size):

        if method == 'he_normal':
            # He Normal: pesos ~ N(0, sqrt(2/n_in))
            # Bom ponto de partida para redes com ReLU — evita gradientes
            # que desaparecem ou explodem desde o início.
            individual = he_normal_ind(n_weights, n_in=n_in)

        elif method == 'he_uniform':
            # He Uniform: pesos ~ Uniform(-sqrt(6/n_in), sqrt(6/n_in))
            # Garante que nenhum peso tem valor extremo logo à partida.
            individual = he_uniform_ind(n_weights, n_in=n_in)

        elif method == 'xavier_normal':
            # Xavier Normal: pesos ~ N(0, sqrt(2/(n_in+n_out)))
            # Recomendado para ativações tanh/sigmoid; equilibra variância
            # no forward e no backward pass.
            individual = xavier_normal_ind(n_weights, n_in=n_in, n_out=n_out)

        else:
            # Random: pesos ~ Uniform(-0.05, 0.05)
            # Método mais simples — funciona bem como baseline.
            individual = np.array(generate_random_solution(n_weights))

        population.append(individual)

    return population
