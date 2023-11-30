# AeroGA

 **Single-objective optimization heuristic algorithm using evolutionary concepts.**

 To use the GA, you should clone the repository, access the folder through the terminal, and perform the installation as shown below:

 ~~~python
 pip install -e .                             
 ~~~

 To call the library just add the commands shown below:

 ~~~python
 from AeroGA.AeroGA import optimize                           
 ~~~

# Steps

### **1. Input Variables**

*Variables related to GA methods*

* **selection** - Selection Method ("roulette", "rank", "tournament").
* **crossover** - Crossover Method ("SBX, ""arithmetic", "1-point", "2-point").
* **mutation** - Mutation Method("gaussian", "polynomial").
* **n_threads** - Number of Threads of the processor that should be used to calculate the fitness function (To use as many threads as possible use -1 as input).

*Variables related to the problem to be solved*

* **lb** - Lower Bounds.
* **ub** - Upper Bounds.
* **nvar** - Number of problem variables.
* **num_generations** - Number of generation.
* **elite** - "global" or "local", local always advances the best of the generation, global the best of the entire optimization so far.
* **elite_count** - Number of individuals that will be passed on to the next generation by elitist means.
* **fitness_fn** - Fitness Function

**Obs.:** To set integer values in GA one must use in lb and ub values such as [0,4], for continuous values one must use [0.0, 4.0].

If the main mdo function is in a different file than the one where the optimization will be done, simply import the function from the file as shown in the example below:

~~~python
from MDO2023_albatroz import MDO2023
~~~ 

To work properly, the fitness function must receive a list X with the values of the individuals, inside the function this list must be opened and assigned to the respective variables.

To perform the optimization one should call the AeroGA function 'optimize', as shown below:

~~~python
out = AeroGA.optimize(selection = "tournament", crossover = "1-point", mutation = "gaussian", n_threads = -1,
    min_values = list, max_values = list, num_variables = int, num_generations = int, elite_count = int, elite="local",
    online_control = False, mutation_prob = 0.4, crossover_prob = 1,
    plotfit = True, plotbox = False, plotparallel = False, 
    fitness_fn = None                                                  
~~~

Some variables already have defaults, so if you don't want to change the defaults they don't need to be set when calling the optimize function.

The *out* dictionary returns the following values:

 * **history** - History of all individuals used in GA
 * **history_valid** - History of all individuals used in GA that are valid, i.e. score != 1000
 * **best_individual** - Best Individual Found
 * **values_gen** - List with values of best fitness, average fitness and metrics found by generation

### **2. Population Initialization**

 Initial stage of the code, where individuals with random values are generated and for each one the fitness value is calculated.

### **3. Selection Criteria**

 Selection criteria are used to select the individuals that will be used in crossover and mutation. The criteria are made so that any individual can be chosen, but those with the highest fitness, have the highest probability of being chosen to generate children. This aspect is important because there is the possibility of individuals with low fitness to maintain the diversity of the population, not discarding regions of the search space. If only the best individuals move forward in the process the convergence becomes fast and the chances of the result falling into a local maximum are high.

 * **Roulette** - In this method, each individual in the population is represented on the roulette wheel in proportion to fitness. Thus, individuals with high fitness are given a larger portion of the roulette wheel, while those with low fitness are given a relatively smaller portion of the wheel. Finally, the roulette wheel is spun a certain number of times, depending on the size of the population, and those drawn on the wheel are chosen as breeding individuals.

 ![Img roleta](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAY_WNtrc6cvHKgM4zkftExoqFzNLrCyMZYSnEDCwYnkSQ8UJhtGJ-mxXUriUOQ3HjVeM&usqp=CAU)

 * **Ranking** - A method similar to roulette, the only difference is that instead of the roulette portion being given by the fitness value, the fitness percentage is considered in relation to the sum of all values. In this way, the ranking method is more democratic and gives more chances to individuals with lower fitness.

 * **Tournament** - The tournament method aeatorically selects two individuals and holds a tournament between them, the winner is the one with the higher fitness value. This is the method best suited to preserve the diversity of the genetic algorithm.

 ![Img torneio](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsy5-dHawPSJtOWkJ9pZtix7tMyHV12N5vdeqc_i9sKOPUE8A7xaN-sl42xTW4Ruxz8w&usqp=CAU)

### **4. Crossover Criteria**

 The recombination operator is the mechanism for obtaining new individuals by exchanging or combining the alleles of two or more individuals. Fragments of the characteristics of an individual are exchanged with an equivalent fragment from another individual. The result of this operation is an individual combining potentially better characteristics of the parents.

 ![Img 1point crossover](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfOL1-3odABVrqx_7KmXVLAbbHupJzM70gHQ&usqp=CAU)

 * **SBX** - [Kalyanmoy et al., 2012](https://content.wolfram.com/uploads/sites/13/2018/02/09-6-1.pdf)

 * **Arithmetic** - Arithmetic recombination creates new alleles in the offspring with values intermediate to those found in the parents. A linear combination is defined between two chromosomes x and y, in order to generate an offspring z.

 * **1 point** - At the 1-point recombination a cut-off point is randomly selected on chromosomes, splitting this into a partition on the right and a partition on the left of the cut-off. Each offspring is composed by joining the left (right) partition of one parent with the right (left) partition of the other parent.

 * **2 points** - The 2-point recombination has the same idea as the 1-point recombination, but two cut-off points are randomly chosen on the chromosomes, dividing the chromosome into three partitions.

### **5. Mutation Criteria**

 The mutation operator randomly changes one or more genes of a chromosome. With this operator an individual generates a copy of itself, which can change. The probability of occurrence of mutation in a gene is called mutation rate. Usually, small values are assigned to the mutation rate, since this operator can generate an individual that is potentially worse than the original.

 The mutation rate has been defined as stochastic and can vary between 5% and 10%. Values larger than this are not recommended as the optimization starts to behave like a random search.

 ![Img mutação](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7eP3cPx5gQXiY04vjPKYHZUc3cyfi98EwPg&usqp=CAU)

 * *Polinomial Mutation* - This mutation follows the same probability distribution as the SBX, where the parameter eta has great influence and refers to the 'strength' of the mutation (Higher values represent lower mutation rates (e.g. 20), lower values represent severe mutation in individuals (e.g. 1)). For more information read the article [(HAMDAN & Mohammad, 2012)](https://d1wqtxts1xzle7.cloudfront.net/31582313/Main-libre.pdf?1392403242=&response-content-disposition=inline%3B+filename%3DThe_Distribution_Index_in_Polynomial_Mut.pdf&Expires=1676061047&Signature=OpI7L7smR9-jq8TBmTeknRwFK83SJz7bnQ0TcQepI4rMvB96v0BSCjhThyORfaaelhAUaSsUlvsLNvNdxlXgPd7UfReDimPBbPtW0RVeeLBWHdjulrTq3JsjqsaGgtRU55fMbAkhe0grDP8uQ2CDsSf8K58YgtikLSWc1lIfIpMGwxfKZodC2IqEOrUaicxh4kNQohiw9T-SjOcpmNKxpW5kYIDjR-lYWr8JfV1yRMDF07HLLf1GMbAgBIw0p47qdPEE0JJG3Q7QBKHtkxxvd7uU2l5g0aBfOoCc4XPQM9u31V2fRkOfXDTQK-h-IEIFqlczRANawigoD6vscTvtgw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

 * *Gaussian Mutation* - In the case of Gaussian mutation, the embedded value in the allele(s) is random with zero mean and standard deviation σ (parameter std_dev). 

 * *Obs.:* Both the parameters std_dev and eta are set stochastically, std_dev can take values between 0.05 and 0.3 and eta values between 10 and 20.
### **6. Quality Metrics**

 * Population diversity - The diversity metric is calculated according to the Euclidean distance between members of the population, the greater the distances the higher the metric and the greater the diversity of the population.
 

### **7. Sensitivity Analysis**

 To import sensitivity function analysis, add the commands shown below:

 ~~~python
 from AeroGA.Utilities.PostProcessing import sensibility                           
 ~~~

 * For this analysis where an individual is used and from the increment the fitness function is calculated by varying each variable of the individual leaving the others fixed.

 ~~~python                                                
 sensibility(individual = list, fitness_fn = None, increment = None, min_values = list, max_values = list)
 ~~~

 The increment can be a fixed continuous value, so all variables will have the same calculation step (except for integer variables, which will always be 1), or you can define a list with specific increment values for each variable.

### **8. Online Parameter Control**

 * The online parameter control serves to vary the mutation rate along the GA, so that the mutation starts high and decreases until it reaches the value input as *mut_prob* in the last generation. This measure is proposed because it is interesting that initially the maximum exploration phase of GA occurs with high mutation and at the end this exploration level is low, allowing GA to develop the individuals found instead of mutating them completely.

### **9. Plots**

 To import plot functions add the commands shown below:

 ~~~python
 from AeroGA.Utilities.Plots import *                           
 ~~~

 * **(BestFit, AvgFit, Metrics) x Generation** - It shows the Fitness score (maximum and average) and metrics over the generations of GA. It can be enabled/disabled in the inputs of the *optimize* function.

 * **Input Dispersion** - Shows the normalized input variables and all points explored by GA during generations. The purpose of this graph is to evaluate how well the algorithm is exploring the search space.

 It can be done in three ways:

 Directly in the optimize function, and will be plotted at the end of the optimization (by default this option is disabled).
 ~~~python
 create_boxplots(out = None, min_values = list, max_values = list)                                       
 ~~~

 After optimization, using GA's results excell.
 ~~~python
 create_boxplots_import_xlsx(path = None)                                       
 ~~~

 After the optimization, using the GA results excell. But here the analysis can be done for a specific generation of the optimization.
 ~~~python
 create_boxplots_por_gen_import_xlsx(path = None, min_values = list, max_values = list, generation = int)                                   
 ~~~

 * **Parallel Curves** - tem o intuito de avaliar a convergência do GA, além de possibilitar a limitação dos bounds.

 It can be done in two ways:

 Directly in the optimize function, and will be plotted at the end of the optimization (by default this option is disabled).
 ~~~python
 parallel_coordinates(out = None)                                      
 ~~~

 After optimization, using GA's results excell.
 ~~~python
  parallel_coordinates_import_xlsx(path = None, classe = None)                                       
 ~~~

 After the optimization, using the GA results excell. But here the analysis can be done for a specific generation of the optimization.
 ~~~python
  parallel_coordinates_per_gen_import_xlsx(path = None, classe = None, generation = int)                                     
 ~~~

 **OBS.:** For the *class* variable, if the input is micro or regular, the variable names for the 2023 project will be used. If you need to change this, you can input *class* as a list containing the new variable names. Ex. *['c1', 'chord_ratio2','b1','span_ratio2','iw','nperfilw1','nperfilw2','zwGround','xCG','vh', 'ih','nperfilh','motorindex']*

# Contact

 Any questions about the code please contact the author.

 Author: Krigor Rosa

 Email: krigorsilva13@gmail.com

# References

 GABRIEL, Paulo Henrique Ribeiro; DELBEM, Alexandre Cláudio Botazzo. Fundamentos de algoritmos evolutivos. 2008.

 VON ZUBEN, Fernando J. Computação evolutiva: uma abordagem pragmática. Anais da I Jornada de Estudos em Computação de Piracicaba e Região (1a JECOMP), v. 1, p. 25-45, 2000.

 HAMDAN, Mohammad. The distribution index in polynomial mutation for evolutionary multiobjective optimisation algorithms: An experimental study. In: International Conference on Electronics Computer Technology (IEEE, Kanyakumari, India, 2012). 2012.
	
