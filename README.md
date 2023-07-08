# AeroGA

**Algoritmo heurístico de otimização single-objective utilizando conceitos evolutivos.**

Para utilizar o GA deve-se clonar o repositório, acessar a pasta pelo terminal e fazer a instalação como mostrado abaixo:

~~~python
pip install -e .                             
~~~

Para chamar a biblioteca basta adicionar os comandos mostrados abaixo:

~~~python
from AeroGA.AeroGA import *                           
~~~

# Etapas

### **1. Variáveis de Entrada**

*Variáveis relacionadas aos métodos do GA*

* **selection** - Método de seleção ("roulette", "rank", "tournament")
* **crossover** - Método de recombinação ("SBX, ""arithmetic", "1-point", "2-point")
* **mutation** - Método de mutação("gaussian", "polynomial")
* **n_threads** - Número de Threads do processador que devem ser utilizadas para calcular aa função fitness (Para utilizar o máximo de threads possíveis utilizar -1 como input)

*Variáveis relacionadas ao problema a ser resolvido*

* **lb** - Lower Bounds
* **ub** - Upper Bounds
* **nvar** - Número de variáveis do problema
* **num_generations** - Número de gerações
* **elite** - "global" ou "local", local avança sempre o melhor da geração, global o melhor da otimização inteira até o momento
* **elite_count** - Número de indivíduos que serão passados para a próxima geração por meio elitista
* **fitness_fn** - Função fitness

**Obs.:** Para definir os valores inteiros no GA deve-se usar no lb e ub valores como [0,4], para valores contínuos deve-se usar [0.0, 4.0]

Caso a função principal do mdo esteja em um arquivo diferente de onde será feita a otimização, basta importar a função do arquivo como mostrado no exemplo abaixo:

~~~python
from MDO2023_albatroz import MDO2023
~~~ 

Para funcionar corretamente, a função fitness deve receber uma lista X com os valores dos indivíduos, dentro da função essa lista deve ser aberta e atribuída as respectivas variáveis.

Para realizar a otimização deve-se chamar a função 'optimize' do AeroGA, como mostrado abaixo:

~~~python
out = AeroGA.optimize(selection = "tournament", crossover = "1-point", mutation = "gaussian", n_threads = -1,
    min_values = list, max_values = list, num_variables = int, population_size = int, num_generations = int, elite_count = int, elite="local",
    online_control = False, mutation_prob = 0.4, crossover_prob = 1, eta = 20, std_dev = 0.1,
    plotfit = True, plotbox = False, plotparallel = False, 
    fitness_fn = None                                                  
~~~

Algumas variáveis já possuem valores defaults, caso você não queira alterar o valor default elas não precisam ser definidas ao chamar a função optimize.

O dicionário *out* retorna os seguintes valores:

 * **history** - Histórico de todos os indivíduos utilizados no GA
 * **history_valid** - Histórico de todos os indivíduos utilizados no GA que sao válidos, ou seja, pontuação != 1000
 * **best_individual** - Melhor indivíduo encontrado
 * **values_gen** - Lista com valores de best fitness, average fitness e metrics encontrados por geração

### **2. Inicialização da População**

Etapa inicial do código, onde são gerados indivíduos com valores aleatórios e para cada um o valor de fitness é calculado.

### **3. Critérios de Seleção**

 Os critérios de seleção são utilizados para selecionar os indivíduos que serão usados no crossover e mutação. Os critérios são feitos de modo que qualquer indivíduo possa ser escolhido porém aqueles com maior fitness, tem consequentemente a maior probabilidade de serem escolhidos para gerar filhos. Tal aspecto é importante pois havendo a possibilidade de indivíduos com baixo fitness mantém-se a diversidade da população, não descartando regiões do espaço de procura. Caso somente os melhores indivíduos passem adiante no processo a convergência se torna rápida e as chances do resultado cair em um máximo local são altas. 

 * **Roleta** - Neste método, cada indivíduo da população é representado na roleta proporcionalmente ao fitness. Assim, aos indivíduos com alto fitness é dada uma porção maior da roleta, enquanto aos de fitness baixo é dada uma porção relativamente menor da roleta. Finalmente, a roleta é girada um determinado número de vezes, dependendo do tamanho da população, e são escolhidos, como indivíduos reprodutores, aqueles sorteados na roleta.

 ![Img roleta](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAY_WNtrc6cvHKgM4zkftExoqFzNLrCyMZYSnEDCwYnkSQ8UJhtGJ-mxXUriUOQ3HjVeM&usqp=CAU)

 * **Ranking** - Método semelhante a roleta, a única diferença é que, ao invés da porção da roleta ser dada pelo valor do fitness considera-se a porcentagem do fitness em relação a soma de todos os valores. Desse modo, o método de ranking é mais democrático e dá mais chances aos indivíduos com menor fitness.

 * **Torneio** - O método do torneio seleciona aeatóriamente dois indivíduos e realiza um torneio entre eles, o vencedor é aquele com maior valor de fitness. Este é o método mais indicado para preservar a diversidade do algoritmo genético.

 ![Img torneio](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsy5-dHawPSJtOWkJ9pZtix7tMyHV12N5vdeqc_i9sKOPUE8A7xaN-sl42xTW4Ruxz8w&usqp=CAU)

### **4. Recombinação (Crossover)**

 O operador de recombinação é o mecanismo de obtenção de novos indivíduos pela troca ou combinação dos alelos de dois ou mais indivíduos. Fragmentos das características de um indivíduo são trocadas por um fragmento equivalente oriundo de outro indivíduo. O resultado desta operação é um indivíduo que combina características potencialmente melhores dos pais.

 ![Img 1point crossover](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfOL1-3odABVrqx_7KmXVLAbbHupJzM70gHQ&usqp=CAU)

 * **SBX** - [Kalyanmoy et al., 2012](https://content.wolfram.com/uploads/sites/13/2018/02/09-6-1.pdf)

 * **Aritmética** - A recombinação aritmética cria novos alelos nos descendentes com valores intermediários aos encontrados nos pais. Define-se uma combinação linear entre dois cromossomos x e y, de modo a gerar um descendente z.

 * **1 ponto** - Na recombinação de 1 ponto, seleciona-se aleatoriamente um ponto de corte nos cromossomos, dividindo este em uma partição à direita e outra à esquerda do corte. Cada descendente é composto pela junção da partição à esquerda (direira) de um pai com a partição à direita (esquerda) do outro pai.

 * **2 pontos** - A recombinação de 2 pontos tem a mesma ideia da recombinação de 1 ponto, porém são escolhidos aleatoriamente dois pontos de corte nos cromossomos, dividindo o cromossomo em três partições.

### **5. Mutação (Mutation)**

 O operador de mutação modifica aleatoriamente um ou mais genes de um cromossomo. Com esse operador, um indivíduo gera uma cópia de si mesmo, a qual pode sofrer alterações. A probabilidade de ocorrência de mutação em um gene é denominada taxa de mutação. Usualmente, são atribuídos valores pequenos para a taxa de mutação, uma vez que esse operador pode gerar um indivíduo potencialmente pior que o original.

 A taxa de mutação foi definida como estocástica e pode variar entre 5% e 10%. Valores maiores que isso não são recomendados pois a otimização começa a se comportar com uma busca aleatória.

 ![Img mutação](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7eP3cPx5gQXiY04vjPKYHZUc3cyfi98EwPg&usqp=CAU)

 * *Mutação Polinomial* - Essa mutação segue a mesma distribuição de probabilidade do SBX, onde o parâmetro eta tem grande influência e se refere a 'força' da mutação (Valores maiores representam taxas de mutações menores (Ex.: 20), valores menores representam mutação severas no indivíduos (Ex.: 1)). Para maiores informações ler o artigo [(HAMDAN & Mohammad, 2012)](https://d1wqtxts1xzle7.cloudfront.net/31582313/Main-libre.pdf?1392403242=&response-content-disposition=inline%3B+filename%3DThe_Distribution_Index_in_Polynomial_Mut.pdf&Expires=1676061047&Signature=OpI7L7smR9-jq8TBmTeknRwFK83SJz7bnQ0TcQepI4rMvB96v0BSCjhThyORfaaelhAUaSsUlvsLNvNdxlXgPd7UfReDimPBbPtW0RVeeLBWHdjulrTq3JsjqsaGgtRU55fMbAkhe0grDP8uQ2CDsSf8K58YgtikLSWc1lIfIpMGwxfKZodC2IqEOrUaicxh4kNQohiw9T-SjOcpmNKxpW5kYIDjR-lYWr8JfV1yRMDF07HLLf1GMbAgBIw0p47qdPEE0JJG3Q7QBKHtkxxvd7uU2l5g0aBfOoCc4XPQM9u31V2fRkOfXDTQK-h-IEIFqlczRANawigoD6vscTvtgw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

 * *Mutação Gaussiana* - No caso da mutação Gaussiana, o valor incorporado no(s) alelo(s) é aleatório com média zero e desvio padrão σ (parâmetro std_dev).

 * *Obs.:* Ambos os parâmetros std_dev e eta são definidos de forma estocástica, std_dev pode assmuir valores entre 0.05 e 0.3 e eta valores entre 10 e 20.
### **6. Métricas de Qualidade**

 * Diversidade da população - A métrica de diversidade é calculada de acordo com a distância euclidiana entre os membros da população, quanto maior as distâncias maior a métrica e maior é a diversidade da população.
 

### **7. Análise de Sensibilidade**

 * Pode ser feita através da função *sensibility*, onde utiliza-se um indivíduo e a partir do incremento calcula-se a função fitness variando cada variável do indivíduo deixando as outras fixas.

~~~python                                                
sensibility(individual = list, fitness_fn = None, increment = None, min_values = list, max_values = list)
~~~

O incremento pode ser um valor contínuo fixo, assim todas as variáveis terão o mesmo step de cálculo (com excessa das variáveis inteiras, que sempre será de 1), ou pode-se definir uma lista com valores de incremento específicos para cada variável.  

### **8. Controle Online de Parâmetros**

 * O controle online de parâmetros serve para variar ao longo do GA a taxa de mutação, de modo que, a mutação começa alta e diminui até chegar ao valor inputado como *mut_prob* na última geração. Essa medidas é propostas pois é interessante que inicialmente ocorrá a fase de máxima exploração do GA com mutação alta e ao final esse nível de exploração seja baixo, permitindo o GA desenvolver os indivíduos encontrados ao invés de mutalos completamente.

### **9. Plots**

 * **(BestFit, AvgFit, Metrics) x Generation** - Mostra o resultado de Fitness (máximo e médio) e métrica ao longo das gerações do GA.

Pode ser ativado/desativado nos inputs da função *optimize*.

 * **Dispersão dos inputs** - Mostra as variáveis de entrada normalizadas e todos os pontos explorados pelo GA durante as gerações. O intúito desse gráfico é avaliar o quão bem o algoritmo está explorando o espaço de busca.

Pode ser feito de tres formas:

Diretamente na função optimize, e será plotado ao final da otimização (por default essa opção fica desativada).
 ~~~python
 create_boxplots(out = None, min_values = list, max_values = list)                                       
 ~~~

Após a otimização, utilizando o excell de resultados do GA. 
 ~~~python
 create_boxplots_import_xlsx(path = None)                                       
 ~~~

Após a otimização, utilizando o excell de resultados do GA. Porém aqui pode ser feita a análise para uma geração específica da otimização.
 ~~~python
 create_boxplots_por_gen_import_xlsx(path = None, min_values = list, max_values = list, generation = int)                                   
 ~~~

 * **Curvas paralelas** - tem o intuito de avaliar a convergência do GA, além de possibilitar a limitação dos bounds.

Pode ser feito de duas formas:

Diretamente na função optimize, e será plotado ao final da otimização (por default essa opção fica desativada).
 ~~~python
 parallel_coordinates(out = None)                                      
 ~~~

Após a otimização, utilizando o excell de resultados do GA. 
 ~~~python
  parallel_coordinates_import_xlsx(path = None, classe = None)                                       
 ~~~

Após a otimização, utilizando o excell de resultados do GA. Porém aqui pode ser feita a análise para uma geração específica da otimização. 
 ~~~python
  parallel_coordinates_per_gen_import_xlsx(path = None, classe = None, generation = int)                                     
 ~~~

**OBS.:** Para a váriável *classe*, se o input for micro ou regular, serão usados os nomes das variáveis referentes ao projeto de 2023. Caso seja necessário mudar isso, pode inputar *classe* como uma lista contendo os novos nomes as variáveis. Ex. *['c1', 'chord_ratio2','b1','span_ratio2','iw','nperfilw1','nperfilw2','zwGround','xCG','vh', 'ih','nperfilh','motorindex']*

# Contato

Qualquer dúvidas sobre o código favor contatar o autor.

Autor: Krigor Rosa

Email: krigorsilva13@gmail.com

# Referências

GABRIEL, Paulo Henrique Ribeiro; DELBEM, Alexandre Cláudio Botazzo. Fundamentos de algoritmos evolutivos. 2008.

VON ZUBEN, Fernando J. Computação evolutiva: uma abordagem pragmática. Anais da I Jornada de Estudos em Computação de Piracicaba e Região (1a JECOMP), v. 1, p. 25-45, 2000.

HAMDAN, Mohammad. The distribution index in polynomial mutation for evolutionary multiobjective optimisation algorithms: An experimental study. In: International Conference on Electronics Computer Technology (IEEE, Kanyakumari, India, 2012). 2012.
	
