"""
population.py — Inicialização da população para o Algoritmo Genético.

A população é uma lista de indivíduos, onde cada indivíduo é um vetor numpy
com n_weights pesos que serão avaliados pela rede neural.

Métodos de inicialização disponíveis:
  - 'random':        Pesos uniformes em [-0.05, 0.05]. Simples e neutro.
  - 'he_normal':     He Normal    — pesos ~ N(0, sqrt(2/n_in)). Para ReLU.
  - 'he_uniform':    He Uniform   — pesos ~ Uniform(-sqrt(6/n_in), sqrt(6/n_in)).
  - 'xavier_normal': Xavier Normal — pesos ~ N(0, sqrt(2/(n_in+n_out))). Para tanh/sigmoid.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_random_solution
from GA_operators.InitializationMethos import (
    he_normal_initialization,
    he_uniform_initialization,
    xavier_normal_initialization,
)


def initialize_population(pop_size, n_weights, method='random', n_in=21, n_out=1):
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
            individual = he_normal_initialization(n_weights, n_in=n_in)

        elif method == 'he_uniform':
            # He Uniform: pesos ~ Uniform(-sqrt(6/n_in), sqrt(6/n_in))
            # Garante que nenhum peso tem valor extremo logo à partida.
            individual = he_uniform_initialization(n_weights, n_in=n_in)

        elif method == 'xavier_normal':
            # Xavier Normal: pesos ~ N(0, sqrt(2/(n_in+n_out)))
            # Recomendado para ativações tanh/sigmoid; equilibra variância
            # no forward e no backward pass.
            individual = xavier_normal_initialization(n_weights, n_in=n_in, n_out=n_out)

        else:
            # Random: pesos ~ Uniform(-0.05, 0.05)
            # Método mais simples — funciona bem como baseline.
            individual = np.array(generate_random_solution(n_weights))

        population.append(individual)

    return population
