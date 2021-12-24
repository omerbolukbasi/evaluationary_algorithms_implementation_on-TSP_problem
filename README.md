This program is a simulation of EA implementation on the given TSP topology. 


Parameters:
file_name          # Input tsp file name.
iteration_count    # Number of iterations.
population_size    # Number of individuals in the population.
crossover_operator # "OX" for Ordered Crossover Operator, "SCX" Sequential Constructive Crossover.
mutation_operator  # "ISM":Insertion Mutation, "IVM":Inversion Mutation, "SM":Swap Mutation, "RM":Random mutation.
generation_count   # Number of iterations(generations).
m                  # Apply 2-opt operator on randomly selected m individuals from the population. If the value is 0, 2-opt will not be implemented.
n                  # 2-opt operator is applied on the selected individual n times. If the value is 0, 2-opt will not be implemented. 
k                  # Once in every k generations apply the 2-opt operator. If the value is 0, 2-opt will not be implemented.



Usage:
python program.py [file_name] [iteration_count] [population_size] [crossover_operator] [mutation_operator] [generation_count] [m] [n] [k]



Example:
(base) C:\Users> python program.py kroA100.tsp 100 50 OX ISM 15 2 3 5
Execution Start Time: 2021-11-20 22:32:59.652513
Iteration: 0
Iteration: 10
Iteration: 20
Iteration: 30
Iteration: 40
Iteration: 50
Iteration: 60
Iteration: 70
Iteration: 80
Iteration: 90
Execution End Time: 2021-11-20 22:33:20.942060

Output Files:
kroA100_results_at_1000_10000_20000_gen.csv     -> Results at generations 1000,10000 and 20000.
kroA100_best_ranks.csv                          -> Best and average result measured in each generation. 
