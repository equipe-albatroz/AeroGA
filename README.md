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

 ![Img torneio](data:image/png;base64,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)

 * **Torneio** - O método do torneio seleciona aeatóriamente dois indivíduos e realiza um torneio entre eles, o vencedor é aquele com maior valor de fitness. Este é o método mais indicado para preservar a diversidade do algoritmo genético.

 ![Img torneio](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsy5-dHawPSJtOWkJ9pZtix7tMyHV12N5vdeqc_i9sKOPUE8A7xaN-sl42xTW4Ruxz8w&usqp=CAU)

### **4. Recombinação (Crossover)**

 O operador de recombinação é o mecanismo de obtenção de novos indivíduos pela troca ou combinação dos alelos de dois ou mais indivíduos. Fragmentos das características de um indivíduo são trocadas por um fragmento equivalente oriundo de outro indivíduo. O resultado desta operação é um indivíduo que combina características potencialmente melhores dos pais.

 ![Img 1point crossover](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAcwAAABtCAMAAAAbMqFLAAABhlBMVEX///+dw+ap0Y6o0I2cwuWt1pH8+/yhyOy/vbvv7+6Bf4GXlphnaWuJrXGHqm+Os9b/AABWXWVlg51deJCBpGmOsnQAAACkoqX/JSX/iIj/z8//R0f/p6elo6GmzvNYYVNmglSCpsWbw4F1mLWZvoGy3JXl5ObY1tSq1PpviqNTZ0ZLXT8yPklDUziCoW1AUF90kGJbcU73GxLInWtja12wrLJSZYFUanxMYHDoOUOLrMvBk6fMbH9gaHFbcEI2Qi2h4JjTdE+tqsiS0fjrPSlffkq3tnt8eHUiIiKGhoYpMyNFVjotNyV6mGdieVIeJhkVGhInMTpVVVXA7aEsPSC24v84R1P/7e3/tbX/3d2Wbm/gIiR0hoeyXV69R0ibk2F7uHi1gFTAXj1aWFhHZDM/U2sqNj+Rn6EcKxEUIi5GXjcmJiY2NjaDXF9bcXSmT1KxOjuCobGawc18mq7B7//Dl6uAcorbJS1WjqukZHfIo6HWhobaaWm7y8kXMUcDFiL/aWn/mJiq/YtpAAASxElEQVR4nO2d/WPjxJnHkTVaZVlspdZaUNpVrbdE8pvsvbLyqde4d5RrcPy6TgyUttc7ID1TSgvL9e3a0v7nHc2LNDN2knWIw0bo+wu7z86LpI+eF42l4aWXChUqVKhQoUKFChUqVOhu6513392+03vfff3mj6TQV9brj7+zfafv7n/75o+k0FfW668VMHOjAmaOVMDMkQqYOVIBM0cqYOZIBcwcqYCZIxUwc6QCZo5UwMyRCpg5UgHzBdW97YVgbt0pgXmNyTaMVOgCHby8vd7+6eP3f7Ztp5/9fP8XW3fapAf3v+5r9sLqcFCucKodndQES7vLW8r/9fjHv+Qt41OhU/mozg/8y//e/x+h08PTh8JM9Y/X5q4LloffO/i6r9kLq8OmwsvrTjzB0ijzFu8/H//4R7zFbYqduh3e8qN/2f93oZPcLAmdOuLcC3FuRf7+DmHeWtS/kXyzNvVhsyRz8ro9T7A0KgpnUBKYvMU9ETop9Q7fCcHkLCW5KfOTe+O1uZsV3lLaIcy9p68IevrBmkU0fLBuubrThmHWO109zNMHazCR4GXC/0UweQuEif5OLRgmspAmCUxuHAST7URgMiYMkxhwIwSzxA6MYLKWXcJ89UO3xskdjHlD7eHIFpqMP4otTnGkW4LmosE5EcZ1Jw1hYLvdFY6mcloROr31ymaYisLDlKFFZmEmBmIhMGUl7ZTBJG0ITBlHRhYm6iVnMOVS2obCZAbGMJm5dwnzjbfEXNGsCQnGbS+EJrUnpsrJrDqipSUJFq2nCMOM68LAC3jmQoI5kYVO5Zc3wVTscsVWGJiwW81mPVMu2bWyq3Ce6VbKpFMKUyZtMExoh1UQsqQwFZtaMEzFTWohl/NMOLDMwiyVahWX8t0pTDHIl4UE4w7EdFJ+ogJOatXhLZLaAoJFa4gJBlYqVyUYu20LnR5uhGkPm80pvsQ4Zypuw6iTcIs90xuctI0y8g8C0x30BgObhSl7E6OJwyKG6TWHvUYn6URzplcxGo0u9n0Ms9LoTQwMEedMpTOdDNoyC7M37Bkdcji7hcmkihKCWWKFYOIcQWNSAlNihGCiP6TmBCZv0Rr01kzj4cST2YyDzrzEJRi7LQudNsGEDJqKN+iii57AhMfYGfc4mEms80b4miOYyTSebKCzJTCTG7fNwVy0uwsPuyrxTHnQWSzIIZGcWfLkWYWF2agvXMPOYHplo+RNRt5teGZy6PR6UZhyegEJzIwnBxNwMDNRmLQJA5PeExgmA4vAlOXs5sIwZabTRpiLad3zGjhf0TC7aPKeCfOWPBszMKHfyw3iQBim4k07dd4zG9N2V2ZgKu60N2q6iszBbLS9LGfC6zObjLpKekqJN8NCd3hLMBXbddMLWkaJX3Zp6YdgwjN3XXzpM5hAA5xnAk3TeJgg1gABT2AqruvKGUwY7FyXjEw9M2miMJ6ZHIwrK5fCnAgwk8F4mPBS9wZMzoTDdtuzBgez3l70mjg4kJxZ645HTRZm2ah3mlOP8UzYeVZjYMKr2T4ZjhmYJbk3OxneDkwYBmbNacPLYMKTODFw5CAwFXfUNnDhSGECVTd0M4MJTH2+NCzMDuVMSWuF8xbrmTAgDk/aNLkhmL12e2DgWx2fuVKfNY2xx3rmAIp02gxzBD2zOeFhNgTP9OpTG5eUBKbnwbNCF53AtEezwQym2gwmbAMvTjJ16pmGvKgYHExvPCWlMPHMQderTcvZKcFqzJXrt+WZJxMYCUpk9jIKL5MhB7Mku0oF1w8EJlDj1jwyQeaZaqyZ5xELE8RmbMTYksCUFXtqLxTGM/EfRiSGJmeu2DN30RspKUz47wPZm5GMuDlnjoeua1QUFqbSrNPQjAug3qxWwrEFF0BKeSy7s04GMynQlMYJ55l112ufsI8m9mDi9UZMAQQPEg+SwZzVldqMgVmSa4o7HN9KAQTzQNmT8WQEJszpIx4m/EOZOBDxTH/lRBxMZKtmMFH9E6wYz4SOOag3OgoHs6TMOjQqIM+cDLrDisfAnJzADEj8bnM1W5oMYJYiw+ICaDQYDDmY0LtHvcwz4Wm3R4M6G2YTHPUJnzMHo4bLVrNKuT06YXMmnKpJkzqBWYEDd5n7s2Q34EyefBueCWFWeJiJC4meCcvFicLkTDUI/X7kAzZnmg71QwQTmMHc6KsszMZ00sVFelrNeuMR+1AGpzaGQ5nxTKU2bU7wg8ZGmMjZYD+S9IlnQtksTKZsJst5SRhV+OdMmV8BktG4/KIBtJSwJVsBKrEwS3hgFmYyN4nEt+SZnUthKvAmJUUIhqnN5/3VPMDZE3umbxmWyRRAQFVBfK6bgIHZ9BaDCQuz5E3T5IbCLLy1SpN2BjO51OPaoH5xmGWVLeeV0gKdFECpyKMJ0ylbAUoteDkvGyVdAUqbpNVsKrpowBwNWTTI+u0+ZzYWY8PDFQYOsxBmGZ8e9czG1CN3G8mZWqyFfSClMIHqGI7PVLNAii0TtCIWZmfqesaYgQkdc8bdxkkdv+jgCEkLINmDBeMWMAXLV4RZ4mBmlu1gpqZdV7O1wXQ0ZqtZGA6n014GE+aB6Wg0wysfJGeqqh/pTJgFarhqrUKQwVSt1nwecs+ZXnM4m3AFkNyc0HuXVAuT2RBW9inMZO7pkCbaTWFWWFj6xv5qgjxT9mzXTkNdEmYV9PSXwUyqWSgb37Tk0QSGWo17ztRiKMCGWRDH6tpzpsIVQLLL3cawRTK5xy4awGdRm9T16zAb4pIvhClYrvl7Jm9Y/z1TgTCFTmu/Z8IbSbDsfgWI+YmBLOcpdOGfVrMKtVyynJcsqLOLBrBMSleG6AoQ8/MGdsk0mqXLeUr6EwOBSX6n2AhzUOZVO/q4JljaXd5SS9404C3jU7HTUZ23oDcN+DaV04rQqS7O7Z4Kc5d3+KYBXZtlLuhFa7Npk4thblzOE2BmM9FHkzWYmdK12bSJCPPgwTX008fv/2rbPr/6+f4vtu60Sa/u7B2gO/+rySbdu/SvL+FXLbd+XeJar1re6qt4z/F7pv1C/555LeX0vdlXPyw/5FQ+rVc4Q2X8vzW+SeWtjxxBfV20rERDcMqP8rB89LEwd02c++F4rZP4psG1lFOYhw9evVLrTR5drQdrlqsnei7t3cBZ5xTmN1MFzBypgJkjFTBzpDsIc6/4qOEC3UGYT40Hh1/j9C+w7gpM5qOMQ8P4dfH92CbdEZiHvz4mgs9j99/4xDh+u8C5prsCM/3CBj9cP3pqGI+K3CnojsB86T4V+fvB3gfGJ4Vz8rorMNd0AH2zcE1edwTmwdtUqJC9twdR/qSoaQXdEZiwgiXaS1C+DFHexMp0znRHYN47OCS6hwLs08Oc71Ry7/41BGFu3wnCvM5c67rWeX5ivJL/AHuw9uX8c3yS/5vH73+67Uf6n/52/3dbd9q0HcAr16pg9vZy7pWJDkONF9D7QLCEz4Q2//Hav77JW5yl0EnTA2GUH+z/m9AmDmOhTXV97kDs9K2iHL1Ih33hxTEziHzB0qfv2dO3zRKYvCUO+SbS2jtqb0KYQicQCm+twbmFYXxxbgn8cGcwr5Vyvl6JMCX+TcHkggqWvsVfcwSTbQPWYAJV52ECBJPrJIGQmwjDFOd2eMsuYe598i1BT0XDJsv3BK0ZNlnWRlm3XD338u1NMPELgCxM1kJgUgMDkzahMNMmKUw6TAozHZjCBNlUm+Z2eMsuYT56shbkHSF5WC1JaOJ8KNu86mPBYI9cwVD7SBgX6JEwt7SeYJZiVgp+sgEmyF6/Tj0zsQAGJkBNAAszfUc7hZl1IjDTTinMzEI9U00nx3MDlTkaBBNbwM5h6kJIN0MxJmlLMQdZR8K7ysm7/LxFIZ+MZQb7iSomGF0Y2A/F5KatBTJnE0xJC5etCGQwgQr0Vuig64dhQl79ViSpLEw4oQ4YmKoWLfv4m0QME6gO7KSpDEygL5ct1IbAVJ3z5TKsZjBhJ3g0AZ07gQnPo9XX1FuAKQT5TTCF99etI0V4x73e4S0IJm+x+0LugDCFuf1QfDFehAk2w1Sd1WfkKxcKU299/qnhpzCBZM6jeI5vXQITmFVjycF09Dicky/7dfxl/7PPv9QROgJTO69+ZnFhVoNTk/0A6NzhZzH5Cod4Zqh/HpKsvHuYTJDHMLMYT2FmTQhM/M45DzP97pR4ZtaGwOSSmy4kGAKTSTAEJtPpApjnkm/SYZOKUo1aXzxjYarWCpwFrSzMQrtlOBzM5MO2ZwbyXhJmgWqehREP0/JNlQ+zZrzCbkdhRmd+GuIRzFZwFi1vByYT0glMNsEQmFkewDDRFmJ0MxkCU6ZfBVGYStoGw+STG4KpZkmIwMzmJjAB02kzTOt4ZVQZmDDMzg2yWQaBqRkBDL0a45lS+MxaSixMOFIY+lmYhQeg9w3LZGCC/twINSbMJuMt8RfkFGZwvFo5TJgFZmwYxxK5wruFCelEURRgF8UwVXgSVbzpC4aJ2pCLQz2z3pvQ3cowTNmrTOiOZijMyt1er2djmjb6MBfAmXQaD5O5JR1aIvzNLoapBlHUp7tahOhbXji3Ay6BmWxR42B2FKYzd8JQlShM9Ek+TG5MAWRW55Z+7uBPvwlMaNNUFiYIgmN0l6QFUFKVIeBpNataBtkfh1azmqavWM+UqsvgXL8dmGZ07gT0Nk5gAn/et+a6mnomPODjQD/GiR/BlL1OozNgNs2R0V4iRsfLYMK/d8ZdJsxC/zgOAoeBKUlO4OiGxsA0W7oTkGoBwVS1ue4Y+KpvhgmjoWnSFuiCanPnDBwTvLiaNc2zKGQKIDM6nq8MFHYITNh3FfsSUwAB0/eDOVvNwpl8fc7DjEKTeTSB6dn0Y0PLYELa5lm8InuW7Rxm/8w3mTALzoxnX4R9Fqa+9NUQJznqmd6iO2RhlrxefcbBdKeLhceEWXjRln6W3EiY9fW+yoRZsxXTlINgJtdCO/uyejFMWKf0nbClpWE2iYYtKzKww2OYauToc+zwJGf6/pmzJNulJDBhVyOyAjwMKYDgjbWKGM9UrdDRj1EUozDh4ZE4QsNsqDvzSMrCrBrPI1jh3ko1C6+FsQpj1jPhabVIwCEwq/PPAd5chMJUOt1pN9tnJdkAcLCg2/3hMOsOptN6ifNMY3VMHuFpAaQCmtwIzPDLVR8wngk5rM7DS3NmrEc6iY/kgmrVSLfM7NFEUqEhNtnnTNgqDpgCCMZmHQb9OCuAEksUALYAAkGkO2wBBC9HQEtu+miCOjFhlhzfrcCUNCvur3zGM319qZMcTgogLVy1jiPWM71Jbzphq1kFsjXGWc6E/1CrjQlwWs06VmDgepnC9CMMKs2ZsWXN+yxM0A8j8mx3UZhlKjjmwZ3mLYvWVaRgoWuzQOIWDQCzIkCr2c2LBlIGMx1E4hYNJC5n3tqiAZwDBnlYx6cwEweKfWeupWFWQmslYZWD6XmuUcZbViQwFdeYDo0Z2icLw5QV2KbRXmQwYQ4yvzACkw2zAEctChOofpKVMpiwKglVP4ougfk8y3nMA8+Fy3nZylxaAG1czuNgMjt3MnNLdO7bXM5LLqgWWXHY8tMwi+v4KGQfTWJHi+bkPBFMZTK2J0NbYTzT8xbpjnsoZ3Ym9phsv4VhmnqgJZmLgWlGc5PxzGSlRrOOycIUhhnMLa2lXwYzE1j75eKChXauST4W2vFzZhCGOg3yJGfCyBYzYRZe4jDS2EcTpdJsN2pMzkw2xvEaNYWB6fbazY7HPGeqVj+kC2Y0zOr0dDFMaIBHQy4tWTSoQgsueDcXQKxw3uIsyXMm99U+8kzOAmHynRKYnAHDZIVgXjm3w1t2H2aT4EZvJbJokKaCdDkP8IsGJbwb2SXLecl/5HQzERxm0bjcownINqbFYTb5d2oiMFXaaBNM/mt71Q8iX7D0Lf6TfDOByVviUOi09mX/m/8HYfJtQF8SOjn6VXOrd3RtljNcsjZL7elyXtboOdZmz6u8grAVCJbzSGjz+8d/+CNv0VdCp2rY5//+xz/t/1boVF2Jc/fX5p5HgkVf5hpmarnOQvv9vWvoz4//8Ndt+/z1//f/snWnjdrZK+no0YQL8qGYYOLkOZOzoBUgbjMZcW8ynDP5XWD6awlGnBvlTC7BaGtZSYR5Ld2RVy231aMrg7wJPVPIDNaRuN+M+P9aUrxBSdyBJuJHgcntORJMCIQmVgHzQj36W1XnVF31eYMefSk00Z+c1gW1PxYtM9Fw9DdhlGrYWp9bnGoldvqo2DrmQh3cSBbYoDd2NfBNvAqcU5jfTBUwc6QCZo5UwMyRCpg5UgEzRypg5kgFzBypgJkjFTBzpAJmjlTAzJEKmDlSATNHKmDmSAXMHKmAmSO9952/b9/p3df+ceMHUuir695772zf6Z3rdCpUqFChQoUKFbo5/RNykCl12mWUeAAAAABJRU5ErkJggg==)

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
	
