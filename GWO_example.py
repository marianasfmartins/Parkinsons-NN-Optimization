"""
Exemplo de uso do Grey Wolf Optimizer corrigido
Demonstra a otimização de pesos com visualização da atualização
"""

from NatureInspiredAlgorithms.NatureInspiredAlgorithms import GreyWolfOptimizer, initialize_population

# Inicializar a população
print("Inicializando população de wolves...")
populacao, num_wolves, n_weights = initialize_population(
    num_wolves=30,  # 30 wolves na população
    n_weights=None   # Calcula automaticamente baseado na rede neural
)

print(f"População inicializada com {num_wolves} wolves e {n_weights} pesos/parâmetros")

# Executar o GWO com 50 iterações e visualização ativada
print("\nIniciando Grey Wolf Optimizer...\n")
best_solution, fitness_history = GreyWolfOptimizer(
    populacao=populacao,
    num_wolves=num_wolves,
    n_weights=n_weights,
    max_iter=10,        # 10 iterações (aumentar para mais precisão)
    fitness_func=None,  # Usa a função de fitness padrão
    visualize=True      # Mostra os gráficos
)

print("\n" + "="*60)
print("RESULTADOS DO GREY WOLF OPTIMIZER")
print("="*60)
print(f"Melhor Fitness Final: {fitness_history[-1]:.6f}")
print(f"Melhoria Total: {fitness_history[0] - fitness_history[-1]:.6f}")
print(f"Número de Iterações: {len(fitness_history) - 1}")
print(f"Taxa de Convergência: {((fitness_history[0] - fitness_history[-1]) / fitness_history[0] * 100):.2f}%")
print("="*60)

