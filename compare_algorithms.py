import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# 1. Import your core project data and evaluation metrics
from project_data import X_train, X_val, Y_train, Y_val
from utils import fitness_misclassification, run_multiple_times, plot_comparison

# 2. Import your setup operators
from GA_operators.population import initialize_population
from GA_operators.crossover import blend_crossover
from GA_operators.mutators import gaussian_mutation
from GA_operators.selector import tournament_selection
from algorithms import genetic_algorithm_l2, grey_wolf_optimizer_l2

NUM_EVAL_RUNS = 5

ga_params = {
        "population": initialize_population,
        "init_method": "he_uniform",
        "pop_size": 100,
        "n_weights": 2401,
        "fit_func": lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val),
        "selector": tournament_selection,
        "mutator": gaussian_mutation,
        "xover_operator": blend_crossover,
        "p_mut": 0.1,
        "p_xover": 0.8,
        "n_gens": 30,
        "pool_size": 5,
        "mutation_strength": 0.1,
        "n_in": 21,
        "n_out": 2
    }

gwo_params = {
        'population': initialize_population,
        "init_method": "he_normal",
        'num_wolves': 100,
        'n_weights': 2401,
        'max_iter': 30,
        'fitness_func': lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val),
        'visualize': False,
        "n_in": 21,
        "n_out": 2
    }
if __name__ == "__main__":
    print("=================== EXECUTING SYSTEM COMPARISON ===================")
    
    # Run GA through your centralized multi-run utility
    print("\n[Processing] Running Genetic Algorithm Pipeline...")
    avg_hist_ga, final_errors_ga, all_ga_runs = run_multiple_times(
        genetic_algorithm_l2, ga_params, n_runs=NUM_EVAL_RUNS
    )
    
    # Run GWO through your centralized multi-run utility
    print("\n[Processing] Running Grey Wolf Optimizer Pipeline...")
    avg_hist_gwo, final_errors_gwo, all_gwo_runs = run_multiple_times(
        grey_wolf_optimizer_l2, gwo_params, n_runs=NUM_EVAL_RUNS
    )

    # =====================================================================
    # 5. MANN-WHITNEY U TEST (INTEGRATED RESULTS ANALYSIS)
    # =====================================================================
    print("\n==================== STATISTICAL VERDICT ====================")
    print(f"GA Final Sample Errors  (Gen 30): {final_errors_ga}")
    print(f"GWO Final Sample Errors (Iter 30): {final_errors_gwo}")
    
    # Perform the leaderboard ranking test
    u_stat, p_value = mannwhitneyu(final_errors_ga, final_errors_gwo, alternative='two-sided')
    
    print(f"Calculated U-Statistic: {u_stat}")
    print(f"Calculated p-value:     {p_value:.5f}")
    
    print("-------------------------------------------------------------")
    if p_value < 0.05:
        print("CONCLUSION: The difference is STATISTICALLY SIGNIFICANT ($p < 0.05$).")
        print("We safely reject the Null Hypothesis. One optimization strategy architecture")
        print("is structurally superior at navigating this specific neural network landscape.")
    else:
        print("CONCLUSION: The difference is NOT STATISTICALLY SIGNIFICANT ($p > 0.05$).")
        print("We fail to reject the Null Hypothesis. The variance in your performance parameters")
        print("can be accounted for by standard stochastic initialization luck.")
    print("=============================================================")

    # =====================================================================
    # CORREÇÃO: EXTRAIR OS CAMPEÕES REAIS DE DENTRO DE ALL_RUNS
    # =====================================================================
    # Assumindo que all_ga_runs/all_gwo_runs é uma lista de tuplos [(best_ind, history), ...]
    # Vamos encontrar a run que obteve o menor erro final
    
    best_ga_idx = np.argmin(final_errors_ga)
    champion_ga = all_ga_runs[best_ga_idx][0]  # Extrai o vetor de pesos campeão do GA
    
    best_gwo_idx = np.argmin(final_errors_gwo)
    champion_gwo = all_gwo_runs[best_gwo_idx][0]  # Extrai o vetor de pesos campeão do GWO

    # =====================================================================
    # 6. UNIFIED CONVERGENCE VISUALIZATION
    # =====================================================================
    # Passamos as histórias médias e os vetores de pesos campeões para o gráfico
    plot_comparison(avg_hist_ga, champion_ga, avg_hist_gwo, champion_gwo)