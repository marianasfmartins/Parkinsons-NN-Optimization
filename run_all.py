"""
run_all.py — Dashboard para correr GWO e GA (duas combinações) e ver os resultados.

Como usar:
    python run_all.py

O que faz (por ordem):
    1. Diagnóstico da rede neural — confirma os shapes dos pesos
    2. Corre o GWO (Grey Wolf Optimizer)
    3. Corre o GA — Combinação 1: Crossover Aritmético + Mutação Gaussiana
    4. Corre o GA — Combinação 2: Blend Crossover + Mutação Polinomial
    5. Comparação final entre os três algoritmos com gráficos

Ficheiros de saída:
    gwo_results.png         — evolução do fitness do GWO
    ga_results_*.png        — evolução do fitness de cada combinação GA
    comparison_results.png  — comparação GWO vs GA1 vs GA2
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NN import get_predictions, get_n_weights, X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_function


# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO DA REDE NEURAL
#  Verifica os shapes reais dos pesos para confirmar que a fórmula do grupo
#  está correta. Se houver discrepância, usa o valor real para evitar erros.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  DIAGNÓSTICO DA REDE NEURAL")
print("═" * 60)

# Treinar 1 iteração apenas para criar a estrutura interna da rede
_nn_diag = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, random_state=42)
_nn_diag.fit(X_train, Y_train)

print(f"  Shapes dos coefs_:      {[c.shape for c in _nn_diag.coefs_]}")
print(f"  Shapes dos intercepts_: {[b.shape for b in _nn_diag.intercepts_]}")

# Comparar a fórmula do grupo com a estrutura real da rede
n_weights      = get_n_weights(X_train)
n_weights_real = (sum(c.size for c in _nn_diag.coefs_)
                  + sum(b.size for b in _nn_diag.intercepts_))

print(f"  n_weights (fórmula grupo): {n_weights}")
print(f"  n_weights (rede real):     {n_weights_real}")

if n_weights != n_weights_real:
    print("\n  ⚠️  AVISO: A fórmula do grupo não bate com a rede real!")
    print(f"     Usando n_weights = {n_weights_real} (rede real) para evitar erros.")
    n_weights = n_weights_real
else:
    print("\n  ✅ Fórmula do grupo está correta — tudo consistente.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FITNESS WRAPPER (partilhado pelos três algoritmos)
#  Esta função é o "árbitro" — avalia um vetor de pesos e devolve o erro.
# ═══════════════════════════════════════════════════════════════════════════════

def nn_fitness_wrapper(pesos):
    """
    Avalia um vetor de pesos (indivíduo/lobo/partícula) usando a rede neural.
    Devolve o erro — quanto menor, melhor.
    """
    previsoes = get_predictions(pesos, X_train, Y_train, X_val, Y_val)
    try:
        valores_reais = Y_val.values
    except AttributeError:
        valores_reais = Y_val
    return fitness_function(previsoes, valores_reais)


# ═══════════════════════════════════════════════════════════════════════════════
#  GWO — GREY WOLF OPTIMIZER
#  Algoritmo bio-inspirado que imita a hierarquia e caça dos lobos cinzentos.
#  Alpha (melhor), Beta (2º), Delta (3º) guiam o resto da alcateia.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  GWO — GREY WOLF OPTIMIZER")
print("═" * 60)

from NatureInspiredAlgorithms.GreyWolfOptimizer import (
    initialize_population as gwo_init_pop,
    grey_wolf_optimizer,
)

GWO_NUM_WOLVES = 30
GWO_MAX_ITER   = 20

print(f"  Lobos: {GWO_NUM_WOLVES} | Iterações: {GWO_MAX_ITER} | Pesos: {n_weights}\n")

gwo_population, _, _ = gwo_init_pop(num_wolves=GWO_NUM_WOLVES, n_weights=n_weights)

gwo_best, gwo_history = grey_wolf_optimizer(
    gwo_population, GWO_NUM_WOLVES, n_weights,
    max_iter     = GWO_MAX_ITER,
    fitness_func = nn_fitness_wrapper,
    visualize    = True,
)

print(f"\n  ✅ GWO concluído — Melhor fitness: {gwo_history[-1]:.8f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GA — IMPORTS E FUNÇÕES DE OPERADORES
# ═══════════════════════════════════════════════════════════════════════════════

from GA_operators.population import initialize_population as ga_init_pop

# Importar via importlib porque o ficheiro tem '+' no nome
cm = importlib.import_module('GA_operators.Crossover+Mutation')
arithmetic_crossover = cm.arithmetic_crossover
gaussian_mutation    = cm.gaussian_mutation
blend_crossover      = cm.blend_crossover
polynomial_mutation  = cm.polynomial_mutation
tournament_selection = cm.tournament_selection


def genetic_algorithm(population, crossover_func, mutation_func,
                      max_iter=20, mutation_rate=0.1, mutation_strength=0.05,
                      pool_size=3, label="GA"):
    """
    Loop principal do Algoritmo Genético.
    Aceita qualquer combinação de crossover e mutação.
    """
    pop_size  = len(population)
    fitnesses = [nn_fitness_wrapper(ind) for ind in population]

    best_fitness    = min(fitnesses)
    best_individual = population[int(np.argmin(fitnesses))].copy()
    history         = [best_fitness]

    print(f"  [{label}] Geração 0/{max_iter} — Fitness inicial: {best_fitness:.6f}")

    for gen in range(max_iter):
        new_population = []

        while len(new_population) < pop_size:
            # Seleção: torneio entre pool_size candidatos aleatórios
            parent1 = tournament_selection(population, fitnesses, pool_size)
            parent2 = tournament_selection(population, fitnesses, pool_size)

            # Crossover: combinar os dois pais
            child1, child2 = crossover_func(parent1, parent2)

            # Mutação: perturbar os filhos aleatoriamente
            child1 = mutation_func(child1, mutation_rate, mutation_strength)
            child2 = mutation_func(child2, mutation_rate, mutation_strength)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population
        fitnesses  = [nn_fitness_wrapper(ind) for ind in population]

        gen_best = min(fitnesses)
        if gen_best < best_fitness:
            best_fitness    = gen_best
            best_individual = population[int(np.argmin(fitnesses))].copy()

        history.append(best_fitness)
        print(f"  [{label}] Geração {gen + 1}/{max_iter} — Fitness: {best_fitness:.6f}")

    return best_individual, history


# ── Wrapper para polynomial_mutation (adapta a assinatura) ───────────────────
def poly_mutation_wrapper(child, mutation_rate, mutation_strength):
    """
    A mutação polinomial precisa de limites (lower/upper bound).
    Este wrapper fixa esses limites e adapta a assinatura à usada pelo GA.
    """
    return polynomial_mutation(child, mutation_rate,
                               lower_bound=-2.0, upper_bound=2.0, eta=20)


# ═══════════════════════════════════════════════════════════════════════════════
#  GA — COMBINAÇÃO 1: Crossover Aritmético + Mutação Gaussiana
#  Mais estável e focado em exploitação.
#  Filhos são interpolações dos pais; mutação adiciona ruído gaussiano suave.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  GA — Combinação 1: Aritmético + Gaussiana | Init: He Normal")
print("═" * 60)

GA_POP_SIZE = 50
GA_MAX_ITER = 20

print(f"  População: {GA_POP_SIZE} | Gerações: {GA_MAX_ITER} | Pesos: {n_weights}\n")

population1 = ga_init_pop(GA_POP_SIZE, n_weights, method='he_normal')

ga1_best, ga1_history = genetic_algorithm(
    population1,
    crossover_func    = arithmetic_crossover,
    mutation_func     = gaussian_mutation,
    max_iter          = GA_MAX_ITER,
    mutation_rate     = 0.1,
    mutation_strength = 0.05,
    pool_size         = 3,
    label             = "Aritmético+Gaussiana+HeNormal",
)

print(f"\n  ✅ GA Comb. 1 concluído — Melhor fitness: {ga1_history[-1]:.8f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GA — COMBINAÇÃO 2: Blend Crossover + Mutação Polinomial
#  Mais exploratório.
#  Filhos podem cair fora do intervalo dos pais; mutação respeita limites fixos.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  GA — Combinação 2: Blend + Polinomial | Init: He Uniform")
print("═" * 60)

print(f"  População: {GA_POP_SIZE} | Gerações: {GA_MAX_ITER} | Pesos: {n_weights}\n")

population2 = ga_init_pop(GA_POP_SIZE, n_weights, method='he_uniform')

ga2_best, ga2_history = genetic_algorithm(
    population2,
    crossover_func    = blend_crossover,
    mutation_func     = poly_mutation_wrapper,
    max_iter          = GA_MAX_ITER,
    mutation_rate     = 0.1,
    mutation_strength = 0.05,
    pool_size         = 3,
    label             = "Blend+Polinomial+HeUniform",
)

print(f"\n  ✅ GA Comb. 2 concluído — Melhor fitness: {ga2_history[-1]:.8f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARAÇÃO FINAL — GWO vs GA1 vs GA2
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  COMPARAÇÃO FINAL")
print("═" * 60)
print(f"  GWO  (Grey Wolf):         {gwo_history[-1]:.8f}")
print(f"  GA 1 (Aritmético+Gauss+HeNormal):  {ga1_history[-1]:.8f}")
print(f"  GA 2 (Blend+Polinomial+HeUniform): {ga2_history[-1]:.8f}")

results = {
    "GWO":                     gwo_history[-1],
    "GA Aritmético+He Normal": ga1_history[-1],
    "GA Blend+He Uniform":     ga2_history[-1],
}
winner = min(results, key=results.get)
print(f"\n  🏆 Vencedor: {winner}  ({results[winner]:.8f})")
print("═" * 60 + "\n")

# ── Gráfico 1: Evolução do fitness dos três algoritmos ───────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(gwo_history,  'b-o', linewidth=2, markersize=4, label='GWO')
ax.plot(ga1_history,  'g-s', linewidth=2, markersize=4, label='GA — Aritmético + Gaussiana + He Normal')
ax.plot(ga2_history,  'r-^', linewidth=2, markersize=4, label='GA — Blend + Polinomial + He Uniform')
ax.set_xlabel('Iteração / Geração', fontsize=12)
ax.set_ylabel('Melhor Fitness', fontsize=12)
ax.set_title('GWO vs GA — Evolução do Fitness', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
print("  Plot de comparação guardado: comparison_results.png")

# ── Gráfico 2: Barra com o resultado final de cada algoritmo ─────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
labels = list(results.keys())
values = list(results.values())
colors = ['steelblue', 'seagreen', 'tomato']
bars = ax2.bar(labels, values, color=colors, edgecolor='black')
ax2.set_ylabel('Melhor Fitness Final', fontsize=12)
ax2.set_title('Comparação do Resultado Final', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Adicionar o valor em cima de cada barra
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('comparison_bar.png', dpi=300, bbox_inches='tight')
print("  Plot de barras guardado: comparison_bar.png\n")
