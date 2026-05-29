"""
GA main.py — Algoritmo Genético para otimização dos pesos de uma rede neural.

O GA funciona como evolução biológica:
  1. Começa com uma população de soluções aleatórias (indivíduos = vetores de pesos).
  2. Avalia o quão boa é cada solução (fitness = erro da rede neural).
  3. Seleciona os melhores para se "reproduzir" (seleção por torneio).
  4. Combina pais para criar filhos (crossover).
  5. Introduz pequenas mudanças aleatórias (mutação) para não ficar preso em mínimos locais.
  6. Repete durante N gerações, guardando sempre o melhor indivíduo.

Corremos duas combinações de operadores para comparar:
  - Combinação 1: Crossover Aritmético + Mutação Gaussiana (mais estável, exploitação)
  - Combinação 2: Blend Crossover  + Mutação Polinomial  (mais exploratório)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import random

# Adiciona a raiz do projeto ao path para os imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar os dados e a rede neural já partilhados com o GWO
from NN import get_predictions, get_n_weights, X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_function
from GA_operators.population import initialize_population

# O ficheiro chama-se 'Crossover+Mutation.py' (tem '+' no nome),
# por isso não podemos usar "from GA_operators.Crossover+Mutation import ..."
# Usamos importlib para carregar o módulo pelo nome em string.
cm = importlib.import_module('GA_operators.Crossover+Mutation')
arithmetic_crossover = cm.arithmetic_crossover  # Combinação 1 — crossover
gaussian_mutation    = cm.gaussian_mutation     # Combinação 1 — mutação
blend_crossover      = cm.blend_crossover       # Combinação 2 — crossover
polynomial_mutation  = cm.polynomial_mutation   # Combinação 2 — mutação
tournament_selection = cm.tournament_selection  # Seleção (usada nas duas combinações)


# ── FITNESS WRAPPER ───────────────────────────────────────────────────────────

'''def nn_fitness_wrapper(pesos):
    """
    Função que avalia a qualidade de um vetor de pesos (um indivíduo do GA).

    Como funciona:
      1. Injeta os pesos do indivíduo na rede neural.
      2. A rede faz previsões no conjunto de validação.
      3. Calcula o erro entre as previsões e os valores reais.
      4. Retorna esse erro — quanto menor, melhor.

    O GA vai tentar MINIMIZAR este valor.
    """
    # Usar a rede neural com estes pesos para prever a doença de Parkinson
    previsoes = get_predictions(pesos, X_train, Y_train, X_val, Y_val)

    # Y_val pode ser um Series do pandas (tem .values) ou já um array numpy
    try:
        valores_reais = Y_val.values
    except AttributeError:
        valores_reais = Y_val

    # Calcular o erro total entre previsões e valores reais
    return fitness_function(previsoes, valores_reais)
'''

# ── ALGORITMO GENÉTICO ────────────────────────────────────────────────────────

'''def genetic_algorithm(population, crossover_func, mutation_func,
                      max_iter=50, mutation_rate=0.1, mutation_strength=0.05,
                      pool_size=3, label="GA", visualize=True):
    """
    Loop principal do Algoritmo Genético.

    A função é genérica: aceita qualquer combinação de crossover e mutação,
    o que nos permite comparar as duas combinações sem duplicar código.

    Args:
        population:       Lista de indivíduos (arrays numpy com os pesos).
        crossover_func:   Função de crossover a usar (aritmético ou blend).
        mutation_func:    Função de mutação a usar (gaussiana ou polinomial).
        max_iter:         Número de gerações do GA.
        mutation_rate:    Probabilidade de mutar cada gene (0.0 a 1.0).
        mutation_strength: Intensidade da mutação (desvio padrão ou equivalente).
        pool_size:        Número de candidatos no torneio de seleção.
        label:            Nome desta combinação (para o print e gráficos).
        visualize:        Se True, guarda o gráfico de evolução do fitness.

    Returns:
        best_individual:  Melhor vetor de pesos encontrado.
        fitness_history:  Lista com o melhor fitness de cada geração.
    """
    pop_size = len(population)

    # ── PASSO 1: Avaliar a população inicial ──────────────────────────────────
    # Cada indivíduo (vetor de pesos) é avaliado pela rede neural.
    # "fitnesses" é uma lista onde fitnesses[i] = erro do indivíduo i.
    fitnesses = [nn_fitness_wrapper(ind) for ind in population]

    # Guardar o melhor da geração 0 (menor erro = melhor solução)
    best_fitness   = min(fitnesses)
    best_individual = population[int(np.argmin(fitnesses))].copy()
    fitness_history = [best_fitness]

    print(f"  [{label}] Geração 0/{max_iter} — Fitness inicial: {best_fitness:.6f}")

    # ── LOOP PRINCIPAL: repetir por N gerações ────────────────────────────────
    for gen in range(max_iter):
        new_population = []

        # ── PASSO 2: Gerar nova geração ───────────────────────────────────────
        # Criar pop_size filhos para substituir a geração atual.
        while len(new_population) < pop_size:

            # SELEÇÃO: escolher dois pais por torneio.
            # O torneio seleciona aleatoriamente `pool_size` indivíduos e
            # devolve o melhor (menor fitness). Favorece os bons mas não garante
            # sempre o mesmo — mantém diversidade.
            parent1 = tournament_selection(population, fitnesses, pool_size)
            parent2 = tournament_selection(population, fitnesses, pool_size)

            # CROSSOVER: combinar os dois pais para gerar dois filhos.
            # O crossover mistura os "genes" (pesos) dos dois pais,
            # transmitindo características de ambos aos filhos.
            child1, child2 = crossover_func(parent1, parent2)

            # MUTAÇÃO: introduzir pequenas perturbações aleatórias nos filhos.
            # Essencial para escapar de mínimos locais e explorar zonas
            # do espaço de soluções que o crossover não consegue alcançar.
            child1 = mutation_func(child1, mutation_rate, mutation_strength)
            child2 = mutation_func(child2, mutation_rate, mutation_strength)

            # Adicionar os filhos à nova população
            new_population.append(child1)
            if len(new_population) < pop_size:  # evitar ultrapassar pop_size
                new_population.append(child2)

        # A nova geração substitui completamente a anterior (substituição geracional)
        population = new_population

        # ── PASSO 3: Avaliar a nova geração ───────────────────────────────────
        fitnesses = [nn_fitness_wrapper(ind) for ind in population]

        # Verificar se há uma solução melhor do que o atual melhor
        gen_best = min(fitnesses)
        if gen_best < best_fitness:
            best_fitness    = gen_best
            best_individual = population[int(np.argmin(fitnesses))].copy()

        fitness_history.append(best_fitness)
        print(f"  [{label}] Geração {gen + 1}/{max_iter} — Melhor fitness: {best_fitness:.6f}")

    # Guardar gráfico de evolução do fitness
    if visualize:
        plot_results(fitness_history, label)

    return best_individual, fitness_history'''






#GA LOOP ADAPTADO ÀS AULAS

def genetic_algorithm(population,
                      fit_func,
                      selector,
                      mutator,
                      xover_operator,
                      p_mut,
                      p_xover,
                      n_gens,
                      pool_size=3,
                      mutation_strength=0.05):
    """
    Algoritmo Genético Clássico da Aula - Adaptado para Minimização de Pesos de NN.
    """
    pop_size = len(population)
    
    # GERAÇÃO 0: Avaliar a população inicial (Passada de fora para consistência com o GWO)
    pop_fits = [fit_func(ind) for ind in population]

    # Como o problema é de MINIMIZAÇÃO (Erro MAE), o melhor é o menor valor
    best_fitness = min(pop_fits)
    best_individual = population[np.argmin(pop_fits)].copy()
    
    # TASK: Criar o histórico para o plot_results
    fitness_history = [best_fitness]

    print(f"[GA] Geração 0/{n_gens} — Fitness Inicial (Menor Erro): {best_fitness:.6f}")

    # Executar a evolução por N gerações
    for generation in range(n_gens):
        offspring = []

        # Preencher a população de descendentes (offspring)
        while len(offspring) < pop_size:

            # Seleção por torneio de 2 pais (Garante consistência teórica)
            # Passamos o pool_size e forçamos a lógica de minimização dentro do seletor
            parent1 = selector(population=population, fitnesses=pop_fits, pool_size=pool_size)
            parent2 = selector(population=population, fitnesses=pop_fits, pool_size=pool_size)

            # Decisão probabilística: Reprodução ou Crossover (Como na Aula!)
            if random.random() <= p_xover:
                child1, child2 = xover_operator(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutação obrigatória com os hiperparâmetros passados
            child1 = mutator(child1, p_mut=p_mut, mutation_strength=mutation_strength)
            child2 = mutator(child2, p_mut=p_mut, mutation_strength=mutation_strength)

            # Adicionar à população de descendentes garantindo que não ultrapassa o pop_size
            offspring.append(child1)
            if len(offspring) < pop_size:
                offspring.append(child2)

        # Substituição geracional completa
        population = [child for child in offspring]
        pop_fits = [fit_func(ind) for ind in population]

        # Atualizar o melhor global do algoritmo (Minimização)
        gen_best_fitness = min(pop_fits)
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[np.argmin(pop_fits)].copy()

        # Guardar no histórico da TASK
        fitness_history.append(best_fitness)
        print(f"[GA] Geração {generation + 1}/{n_gens} — Melhor Fitness: {best_fitness:.6f}")

    # Retorna o melhor indivíduo (pesos ótimos) e o histórico para o gráfico comparativo
    return best_individual, fitness_history

# ── PLOT ──────────────────────────────────────────────────────────────────────

def plot_results(fitness_history, label="GA"):
    """Guarda o gráfico de evolução do fitness ao longo das gerações."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fitness_history, 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Geração', fontsize=12)
    ax.set_ylabel('Melhor Fitness', fontsize=12)
    ax.set_title(f'Evolução do Fitness — {label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Nome do ficheiro baseado no label (sem espaços)
    filename = f"ga_results_{label.replace(' ', '_').replace('+', 'e')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Plot guardado: {filename}")
    plt.close()

#BAYESIAN OPTIMIZATION SEARCH
def objective(trial):
    # 1. Sugerir o método de inicialização
    init_method = trial.suggest_categorical('init_method', ['random', 'he_normal', 'he_uniform', 'xavier_normal'])
    
    # 2. Sugerir os operadores do GA
    xover_op = trial.suggest_categorical('xover', [arithmetic_crossover, blend_crossover])
    mut_op = trial.suggest_categorical('mutation', [gaussian_mutation, polynomial_mutation])
    
    # 3. Gerar a população usando o método sugerido pelo trial
    # (Nota: n_weights para o teu MLP com hidden_layer_sizes=(100,) e 21 features de entrada seria 100*21 + 100 + 100*2 + 2 = 2402 se considerarmos pesos e biases!)
    population = initialize_population(pop_size=30, n_weights=2402, method=init_method, n_in=21, n_out=2)
    fitness_func = lambda ind: fitness_function(ind, X_train, Y_train, X_val, Y_val)
    
    # 4. Executar o GA
    best_weights, ga_hist = genetic_algorithm(
        population=population,
        fit_func=fitness_func,
        selector=tournament_selection,
        mutator=mut_op,
        xover_operator=xover_op,
        p_mut=trial.suggest_float('p_mut', 0.01, 0.2),
        p_xover=trial.suggest_float('p_xover', 0.6, 0.9),
        n_gens=30
    )
    return min(ga_hist)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    '''print("\n" + "=" * 55)
    print("  GA — GENETIC ALGORITHM (duas combinações)")
    print("=" * 55)

    # Calcular o número de pesos a otimizar com base nas features do dataset
    n_weights = get_n_weights(X_train)
    print(f"  Total de pesos a otimizar: {n_weights}")

    pop_size = 50
    max_iter = 50

    # ── COMBINAÇÃO 1: Aritmético + Gaussiana + He Normal ─────────────────────
    # Crossover aritmético: filhos são interpolações lineares dos pais (exploitação).
    # Mutação gaussiana: adiciona ruído com distribuição normal (exploração suave).
    # Inicialização He Normal: pesos ~ N(0, sqrt(2/n_in)) — ideal para ReLU.
    print(f"\n  Combinação 1: Aritmético + Gaussiana | Init: He Normal")
    print(f"  Inicializando população ({pop_size} indivíduos)...")
    population1 = initialize_population(pop_size, n_weights, method='he_normal')

    best1, hist1 = genetic_algorithm(
        population1,
        crossover_func    = arithmetic_crossover,
        mutation_func     = gaussian_mutation,
        max_iter          = max_iter,
        mutation_rate     = 0.1,    # 10% dos genes podem mutar por geração
        mutation_strength = 0.05,   # desvio padrão da perturbação gaussiana
        pool_size         = 3,      # 3 candidatos por torneio
        label             = "Aritmético + Gaussiana + He Normal",
        visualize         = True
    )
    print(f"\n  ✅ Combinação 1 — Melhor fitness: {hist1[-1]:.8f}")

    # ── COMBINAÇÃO 2: Blend + Polinomial + He Uniform ────────────────────────
    # Blend crossover: filhos podem ir ALÉM do intervalo dos pais (mais exploração).
    # Mutação polinomial: perturbação dentro de [-2, 2], controlada pelo parâmetro eta.
    # Inicialização He Uniform: pesos ~ Uniform(-sqrt(6/n_in), sqrt(6/n_in)).
    print(f"\n  Combinação 2: Blend + Polinomial | Init: He Uniform")
    print(f"  Inicializando população ({pop_size} indivíduos)...")
    population2 = initialize_population(pop_size, n_weights, method='he_uniform')

    # A mutação polinomial precisa de limites — criamos um wrapper com a mesma
    # assinatura que a gaussiana para poder usar a mesma função genetic_algorithm.
    def poly_mutation_wrapper(child, mutation_rate, mutation_strength):
        """Adapta polynomial_mutation à assinatura esperada pelo genetic_algorithm."""
        # Usamos mutation_strength como proxy: strength mais alto → eta mais baixo
        # (mais exploração). Aqui fixamos eta=20 (exploitação moderada).
        return polynomial_mutation(child, mutation_rate,
                                   lower_bound=-2.0, upper_bound=2.0, eta=20)

    best2, hist2 = genetic_algorithm(
        population2,
        crossover_func    = blend_crossover,
        mutation_func     = poly_mutation_wrapper,
        max_iter          = max_iter,
        mutation_rate     = 0.1,
        mutation_strength = 0.05,   # não usado pelo poly_mutation_wrapper, mas necessário pela assinatura
        pool_size         = 3,
        label             = "Blend + Polinomial + He Uniform",
        visualize         = True
    )
    print(f"\n  ✅ Combinação 2 — Melhor fitness: {hist2[-1]:.8f}")

    # ── COMPARAÇÃO ENTRE AS DUAS COMBINAÇÕES ──────────────────────────────────
    print("\n" + "=" * 55)
    print("  COMPARAÇÃO — GA Combinação 1 vs Combinação 2")
    print("=" * 55)
    print(f"  Aritmético + Gaussiana + He Normal:   {hist1[-1]:.8f}")
    print(f"  Blend      + Polinomial + He Uniform: {hist2[-1]:.8f}")
    winner = "Aritmético + Gaussiana + He Normal" if hist1[-1] < hist2[-1] else "Blend + Polinomial + He Uniform"
    print(f"  🏆 Melhor combinação: {winner}")
    print("=" * 55)

    # Plot de comparação entre as duas combinações do GA
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist1, 'g-o', linewidth=2, markersize=4, label='Aritmético + Gaussiana + He Normal')
    ax.plot(hist2, 'b-s', linewidth=2, markersize=4, label='Blend + Polinomial + He Uniform')
    ax.set_xlabel('Geração', fontsize=12)
    ax.set_ylabel('Melhor Fitness', fontsize=12)
    ax.set_title('GA — Comparação de Combinações de Operadores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ga_comparison.png', dpi=300, bbox_inches='tight')
    print("\n  Plot de comparação guardado: ga_comparison.png")
'''