# Tic Tac Toe com Aprendizado por Reforço

Este projeto implementa um agente que joga o clássico jogo da velha (Tic Tac Toe) usando aprendizado por reforço. O objetivo é permitir que o agente jogue contra um jogador humano, após ser treinado para otimizar suas decisões com base nas equações de Bellman.

## Como Configurar o Projeto

Siga as etapas abaixo para configurar e rodar o projeto no seu computador.

### 1. Pré-requisitos

Antes de começar, você precisa ter o Python instalado. Se ainda não tiver, baixe-o [aqui](https://www.python.org/downloads/). Recomenda-se ter a versão 3.x.

### 2. Criando um Ambiente Virtual

Para manter o projeto isolado e evitar conflitos de dependências, vamos usar um ambiente virtual:

1. Abra o terminal e vá para a pasta do projeto (onde você descompactou ou criou o arquivo do projeto).
   
2. Crie o ambiente virtual com o seguinte comando:
   ```bash
   python -m venv venv

3. Ative o ambiente virtual:

No Windows
``` bash
.\venv\Scripts\activate
```
No Mac
``` bash
source venv/bin/activate
```

4. Instale o NumPy, que é necessário para manipulação de matrizes no projeto:
``` bash
pip install numpy
```
### 3. Rodando o código

Após configurar o ambiente, copie o arquivo tic_tac_toe_rl.py para a pasta do projeto.

No terminal (com o ambiente virtual ativo), execute o código com:

``` bash
python Jogo-Velha.py
```

### 4. Como Jogar contra o Agente

Você é o 'O' e o agente é o 'X'.

No seu turno, digite as coordenadas de sua jogada no formato:
``` bash
linha coluna
```

Por exemplo, para jogar na primeira linha e segunda coluna, você digitaria:
``` bash
0 1
```

Após o término de uma partida, o jogo perguntará:
``` bash
Deseja jogar novamente? (s/n):
```
Digite 's' para jogar outra vez.
Digite 'n' para encerrar o jogo.

O jogo continuará até um dos jogadores vencer ou empatar.

### 5. Explicação do Código
O código está dividido em duas classes principais: TicTacToeEnv (ambiente do jogo) e RLAgent (agente de aprendizado por reforço).

Estrutura do Ambiente: TicTacToeEnv
- A classe TicTacToeEnv é responsável por modelar o jogo como um Processo de Decisão de Markov (MDP), onde:

- Estado: Representado pelo estado atual do tabuleiro.
- Ações: Jogadas possíveis em cada célula vazia do tabuleiro.
- Recompensa: Definida com base no resultado final da partida (1 para vitória do agente, -1 para derrota e 0 para empate).

TicTacToeEnv (Ambiente do Jogo)
__init__: Inicializa o tabuleiro 3x3 vazio e define o jogador inicial como '1' (agente).
- reset: Reinicia o tabuleiro para um novo jogo.
- is_valid_action: Verifica se uma célula está livre para receber uma jogada.
    - Se a célula estiver vazia (0), a jogada é permitida; caso contrário, é considerada inválida.
- step: Executa uma jogada, atualizando o estado do tabuleiro e alternando o jogador.
    - Se a jogada for válida, atualiza o estado do tabuleiro, troca o jogador e verifica se há um vencedor.
    - Se a jogada for inválida, penaliza com uma recompensa de -1.
- check_winner: Verifica se há um vencedor ou empate.
    - Para isso, checa linhas, colunas e diagonais. Se o tabuleiro estiver cheio sem um vencedor, considera-se empate.
- render: Exibe o estado atual do tabuleiro de maneira visual, mostrando 'X' para o agente, 'O' para o jogador humano e ' ' para células vazias.

* Nota: Essa classe modela o jogo da velha como um Processo de Decisão de Markov (MDP), onde o estado é o tabuleiro, as ações são as jogadas válidas, e a recompensa está associada ao resultado final (vitória, derrota ou empate).

#### RLAgent (Agente de Aprendizado por Reforço):
#####  O agente utiliza a técnica de Q-Learning, que é baseada na equação de Bellman para atualizar sua função de valor a cada jogada, com o objetivo de melhorar sua política de jogo.

- __init__: Inicializa o agente com uma função de valor vazia e define os hiperparâmetros (epsilon, alpha, gamma).
- epsilon: Probabilidade de explorar novas ações aleatórias durante o treinamento.
- alpha: Taxa de aprendizado, que define o quão rápido o agente se ajusta às novas recompensas.
- gamma: Fator de desconto, que prioriza recompensas futuras.
- get_state_key: Converte o estado do tabuleiro em uma tupla para ser usada como chave no dicionário da função de valor.
- choose_action: Implementa a política epsilon-greedy, escolhendo ações com base na função de valor ou aleatoriamente (exploração).
- update_value_function: Atualiza a função de valor usando a equação de Bellman, considerando a recompensa obtida.
- train: Treina o agente em várias partidas de simulação, ajustando sua política com base nas recompensas recebidas.
- play_against_human: Permite que um jogador humano jogue contra o agente após o treinamento.

#### Comentários sobre o Aprendizado do Agente:
- O agente foi treinado usando aprendizado por reforço com decaimento do epsilon, o que ajuda a torná-lo mais adaptativo ao longo do tempo.
- A implementação atual usa Q-Learning, uma técnica baseada na atualização da função de valor usando a equação de Bellman:
``` bash
new_value = old_value + alpha * (reward + gamma * next_value - old_value)
```
- O decaimento do epsilon reduz gradualmente a exploração, permitindo que o agente se concentre em estratégias mais conhecidas à medida que aprende.
- A função de valor é ajustada com base nas recompensas que o agente recebe durante as partidas simuladas.
- A exploração é controlada pelo epsilon, permitindo que o agente teste novas estratégias em busca de melhores resultados.

#### Considerações sobre o Aprendizado do Agente
- Q-Learning: A técnica usada para treinar o agente é baseada na equação de Bellman, que permite ao agente ajustar sua função de valor a cada passo do jogo.
- Decaimento do Epsilon: Ajuda o agente a explorar menos ao longo do tempo, focando mais em estratégias conhecidas e aprendidas.
- Hiperparâmetros: Os valores de epsilon, alpha e gamma foram ajustados para permitir um aprendizado eficiente, enquanto o decaimento do epsilon ajuda a transitar de uma fase de exploração para uma fase de exploração reduzida.