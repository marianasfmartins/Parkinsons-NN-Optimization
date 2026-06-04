from project_data import X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_misclassification, plot_history
from algorithms import grey_wolf_optimizer, grey_wolf_optimizer_l2
from GA_operators.population import initialize_population
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    gwo_params = {
        'population': initialize_population,
        "init_method": "he_normal",
        'num_wolves': 50,
        'n_weights': 2401,
        'max_iter': 30,
        'fitness_func': lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val),
        'visualize': False,
        "n_in": 21,
        "n_out": 2
    }
    '''best_wolf, fitness_history = grey_wolf_optimizer(**gwo_params)
    print("Best Individual fitness:", fitness_history[-1])
    plot_history(fitness_history, title="GWO Fitness Evolution")
    plt.savefig("GWO_Fitness_Evolution.png")
    plt.show()
'''
    print("=================== STARTING CHAMPION DEPLOYMENT ===================")
    print("[Processing] Running the GWO to isolate the optimal weights...")
        
    # Execute your updated function
    # It will now handle the L2 tie-breaking internally at each generation!
    champion_weights, fitness_history = grey_wolf_optimizer_l2(**gwo_params)
        
    # Calculate the final L2 Norm of the champion for the report
    final_l2_norm = np.sum(np.array(champion_weights) ** 2)
        
    print("\n=================== OPTIMIZATION COMPLETE ===================")
    print(f"--> Best Validation Error Achieved: {fitness_history[-1]} errors")
    print(f"--> Model Structural Complexity (L2 Norm): {final_l2_norm:.4f}")
    plot_history(fitness_history, title="GA Fitness Evolution (L2 Norm)", L2=final_l2_norm)
    plt.savefig("GA_Fitness_Evolution_L2.png")
    plt.show()
