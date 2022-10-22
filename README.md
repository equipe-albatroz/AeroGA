# AeroGA

Algoritmo heurístico de otimização single-objective utilizando conceitos evolutivos.

Para utilizar o GA deve-se fazer a importação do arquivo .py e todas suas funções, como mostrado abaixo:

~~~python
import AeroGA                              
~~~

# Etapas

### **1. Carregando variáveis**

As variáveis utilizadas no GA são repassadas através de uma estrutura. Para criar a estrutura é necessário importar a bibliotéca ypstruct, como mostrado abaixo:

~~~python
from ypstruct import structure                            
~~~

Para instalar a bibliotéca ypstruct e todas as outras necessárias para rodar o código siga os passos mostrados na seção 'Instalando requirements' em Observações.

Variáveis relacionadas ao problema a ser resolvido

* fitness - Função que deve ser otimizada
* nvar - Número de variáveis do problema
* lb - Lower Bounds
* ub - Upper Bounds
* integer - Indice de números inteiros

Devem ser definidas como no exemplo abaixo:

~~~python
problem = structure()                                 
problem.fitness = sphere                         
problem.nvar = 5                                     
problem.lb = [0.2, -10, -10, -5, -5]                 
problem.ub = [0.4 , 10, 10,  5, 5]                   
problem.integer = [1,2]                                
~~~

Variáveis relacionadas aos parâmetros do GA

* max_iterations - Número máximo de iterações
* npop - Tamanho da população
* pc - Proporção da população de filhos em relação aos pais
* mu - Taxa de mutação(pode ser declarado um valor geral, ou, um vetor contendo a taxa para cada variável como no exemplo)
* sigma - Desvio padrão da mutação para números contínuos
* sigma_int - Desvio padrão da mutação para números inteiros
* gamma - Amplitude da recombinação aritmética
* elitism - Porcetagem da população que será elitizada (melhores indivíduos passam para a próxima geração)

Devem ser definidas como no exemplo abaixo:

~~~python
params = structure()                           
params.max_iterations = 100                     
params.npop = 50                            
params.pc = 1                                 
params.mu = [0.5, 0.5, 0.5, 0.5, 0.5]           
params.sigma = 0.1                               
params.sigma_int = 0.7                       
params.gamma = 0.1                            
params.elitism = 0.1                                                 
~~~

Variáveis relacionadas aos parâmetros do GA

* selection - Método de seleção ("roulette", "rank", "tournament")
* crossover - Método de recombinação ("arithmetic", "1-point", "2-point")
* mutation - Método de mutação("gaussian", "default")

Devem ser definidas como no exemplo abaixo:

~~~python
methods = structure()
methods.selection = "tournament"    
methods.crossover = "arithmetic"                  
methods.mutation = "gaussian"                                                          
~~~

Para realizar a otimização deve-se chamar a função 'optimize' do AeroGA, como mostrado abaixo:

~~~python
outputs = AeroGA.optimize(problem, params, methods)                                                  
~~~

### **2. Inicialização da População**

Etapa inicial do código, onde são gerados indivíduos com valores aleatórios e para cada um o valor de fitness é calculado.

### **3. Critérios de Seleção**

Os critérios de seleção são utilizados para selecionar os indivíduos que serão usados no crossover e mutação. Os critérios são feitos de modo que qualquer indivíduo possa ser escolhido porém aqueles com maior fitness, tem consequentemente a maior probabilidade de serem escolhidos para gerar filhos. Tal aspecto é importante pois havendo a possibilidade de indivíduos com baixo fitness mantém-se a diversidade da população, não descartando regiões do espaço de procura. Caso somente os melhores indivíduos passem adiante no processo a convergência se torna rápida e as chances do resultado cair em um máximo local são altas. 

 * **Roleta** - Neste método, cada indivíduo da população é representado na roleta proporcionalmente ao fitness. Assim, aos indivíduos com alto fitness é dada uma porção maior da roleta, enquanto aos de fitness baixo é dada uma porção relativamente menor da roleta. Finalmente, a roleta é girada um determinado número de vezes, dependendo do tamanho da população, e são escolhidos, como indivíduos reprodutores, aqueles sorteados na roleta.

![Img roleta](img/roleta.png)

 * **Ranking** - Método semelhante a roleta, a única diferença é que, ao invés da porção da roleta ser dada pelo valor do fitness considera-se a porcentagem do fitness em relação a soma de todos os valores. Desse modo, o método de ranking é mais democrático e dá mais chances aos indivíduos com menor fitness.

 * **Torneio** - O método do torneio seleciona aeatóriamente dois indivíduos e realiza um torneio entre eles, o vencedor é aquele com maior valor de fitness. Este é o método mais indicado para preservar a diversidade do algoritmo genético.

 ![Img torneio](img/torneio.jpg)

### **4. Recombinação (Crossover)**

 * **Aritmética** - A recombinação aritmética cria novos alelos nos descendentes com valores intermediários aos encontrados nos pais. Define-se uma combinação linear entre dois cromossomos x e y, de modo a gerar um descendente z.

 * **1 ponto** - Na recombinação de 1 ponto, seleciona-se aleatoriamente um ponto de corte nos cromossomos, dividindo este em uma partição à direita e outra à esquerda do corte. Cada descendente é composto pela junção da partição à esquerda (direira) de um pai com a partição à direita (esquerda) do outro pai.

![Img roleta](img/1point.png)

 * **2 pontos** - A recombinação de 2 pontos tem a mesma ideia da recombinação de 1 ponto, porém são escolhidos aleatoriamente dois pontos de corte nos cromossomos, dividindo o cromossomo em três partições.

![Img roleta](img/2point.png)


### **5. Mutação (Mutation)**

 * *Mutação padrão*
 * *Mutação Gaussiana*

### **6. Métricas de Qualidade**
### **7. Plots**




# Observações

### **1. Instalando requirements.txt**

Para instalar todas as bibliotecas utilizadas, no terminal de comandos coloque o código abaixo:

~~~python
pip install -r requirements.txt
~~~

### **2. Paralelização**

Opção de paralelização ainda não foi adicionada, o código AeroGA_parallel está em desenvolvimento.

# Contato

Qualquer dúvidas sobre o código favor contatar o autor.

Author: Krigor Rosa

Email: krigorsilva13@gmail.com
	