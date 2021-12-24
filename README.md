## Evaluationary Algorithms Implementation on Travelling Salesman Problem<br>
This program is a simulation of EA implementation on the given TSP topology. Several genetic mutation and recombination algorithms are applied and being compared within an academic paper. Below is the usage of the program. 
<br>
<br>

#### Parameters:
<br>

**file_name**          : Input tsp file name.<br>
**iteration_count**    : Number of iterations.<br>
**population_size**    : Number of individuals in the population.<br>
**crossover_operator** : "OX" for Ordered Crossover Operator, "SCX" Sequential Constructive Crossover.<br>
**mutation_operator**  : "ISM":Insertion Mutation, "IVM":Inversion Mutation, "SM":Swap Mutation, "RM":Random mutation.<br>
**generation_count**   : Number of iterations(generations).<br>
**m**                  : Apply 2-opt operator on randomly selected m individuals from the population. If the value is 0, 2-opt will not be implemented.<br>
**n**                  : 2-opt operator is applied on the selected individual n times. If the value is 0, 2-opt will not be implemented. <br>
**k**                  : Once in every k generations apply the 2-opt operator. If the value is 0, 2-opt will not be implemented.<br>

<br>
<br>

#### Usage:
<br>
python program.py [file_name] [iteration_count] [population_size] [crossover_operator] [mutation_operator] [generation_count] [m] [n] [k]<br>
<br>


#### Example:<br>
<br>
(base) C:\Users> python program.py kroA100.tsp 100 50 OX ISM 15 2 3 5<br>
Execution Start Time: 2021-11-20 22:32:59.652513<br>
Iteration: 0<br>
Iteration: 10<br>
Iteration: 20<br>
Iteration: 30<br>
Iteration: 40<br>
Iteration: 50<br>
Iteration: 60<br>
Iteration: 70<br>
Iteration: 80<br>
Iteration: 90<br>
Execution End Time: 2021-11-20 22:33:20.942060<br>
<br>

#### Output Files:
<br>
kroA100_results_at_1000_10000_20000_gen.csv     -> Results at generations 1000,10000 and 20000.<br>
kroA100_best_ranks.csv                          -> Best and average result measured in each generation. <br>
<br>
