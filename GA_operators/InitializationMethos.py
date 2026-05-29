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


def he_normal_initialization(n_weights, n_in=21):
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


def he_uniform_initialization(n_weights, n_in=21):
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


def xavier_normal_initialization(n_weights, n_in=21, n_out=1):
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

