import numpy as np
import matplotlib.pyplot as plt
from project_data import X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_misclassification, plot_history
from GA_operators.population import initialize_population
from algorithms import grey_wolf_optimizer_l2
from NN import get_predictions

# =====================================================================
# STEP 1: GREY WOLF OPTIMIZER CONFIGURATION
# =====================================================================
GWO_FINAL_CONFIG = {
    "population": initialize_population,
    "init_method": "he_uniform",        # Consistent baseline initialization
    "num_wolves": 100,                  # Population size mapped to match GA budget
    "n_weights": 2401,                  # Flat neural network weight dimensionality
    "max_iter": 30,                     # Total iteration execution budget
    "fitness_func": lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val),
    "visualize": False,                 # Managed manually via step 3 below
    "n_in": 21,
    "n_out": 2
}

if __name__ == "__main__":
    print("=================== STARTING CHAMPION DEPLOYMENT ===================")
    print("[Processing] Running the GWO to isolate the optimal weights...")
    
    # 1. Execute your updated GWO function
    # It handles the L2 tie-breaking internally across the social hierarchy!
    champion_weights, fitness_history = grey_wolf_optimizer_l2(**GWO_FINAL_CONFIG)
    
    # 2. Calculate the final L2 Norm of the alpha wolf for the report
    final_l2_norm = np.sum(np.array(champion_weights) ** 2)
    
    print("\n=================== OPTIMIZATION COMPLETE ===================")
    print(f"--> Best Validation Error Achieved: {fitness_history[-1]} errors")
    print(f"--> Model Structural Complexity (Alpha L2 Norm): {final_l2_norm:.4f}")

    # =====================================================================
    # STEP 2: UNBIASED TEST SET EVALUATION (THE UNLOCKED VAULT)
    # =====================================================================
    print("\n=================== UNSEEN TEST SET RESULTS ===================")
    print("[Processing] Passing the champion alpha weight vector through the test set...")
    
    # Generate the network's final hard predictions on the unseen test features
    Y_pred_test = get_predictions(champion_weights, X_train, Y_train, X_test, Y_test)
    
    # Ensure both arrays match structures for comparison element-wise
    Y_pred_test = np.array(Y_pred_test)
    Y_test = np.array(Y_test)
    
    # Calculate performance metrics
    test_misclassifications = np.sum(Y_pred_test != Y_test)
    test_accuracy = (1.0 - (test_misclassifications / len(Y_test))) * 100
    
    print("\n=================== FINAL PERFORMANCE METRICS ===================")
    print(f"--> Total Unseen Test Records:      {len(Y_test)}")
    print(f"--> Total Test Misclassifications: {test_misclassifications}")
    print(f"--> Final Generalization Accuracy:  {test_accuracy:.2f}%")
    print("=================================================================")
    
  