import numpy as np
import random

# Ambiente do Jogo da Velha
class TicTacToeEnv:
    def __init__(self):
        """
        Inicializa o tabuleiro 3x3 e define o jogador inicial.
        """
        self.board = np.zeros((3, 3), dtype=int)  # Tabuleiro vazio
        self.current_player = 1  # 1 para o agente ('X'), -1 para o humano ('O')

    def reset(self):
        """
        Limpa o tabuleiro e começa uma nova partida.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Agente começa primeiro
        return self.board

    def is_valid_action(self, row, col):
        """
        Verifica se a célula está livre.
        """
        return self.board[row, col] == 0

    def step(self, row, col):
        """
        Faz a jogada no tabuleiro, se for válida.
        """
        if self.is_valid_action(row, col):
            self.board[row, col] = self.current_player  # Faz a jogada
            reward, done = self.check_winner()  # Vê se alguém ganhou
            self.current_player *= -1  # Troca o jogador
            return self.board, reward, done
        else:
            return self.board, -1, False  # Penaliza jogada inválida

    def check_winner(self):
        """
        Verifica se alguém venceu ou se houve empate.
        """
        for player in [1, -1]:  # Checa para agente ('X') e humano ('O')
            # Confere linhas, colunas e diagonais
            if any(np.all(self.board[i, :] == player) for i in range(3)) or \
               any(np.all(self.board[:, j] == player) for j in range(3)) or \
               np.all(np.diag(self.board) == player) or \
               np.all(np.diag(np.fliplr(self.board)) == player):
                return (1 if player == 1 else -1), True

        # Verifica empate
        if not np.any(self.board == 0):
            return 0, True  # Empate

        return 0, False  # Jogo continua

    def render(self):
        """
        Exibe o estado atual do tabuleiro de forma visual.
        """
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print('|'.join(symbols[cell] for cell in row))
            print('-' * 5)


# Agente de Aprendizado por Reforço
class RLAgent:
    def __init__(self, epsilon=0.9, alpha=0.5, gamma=0.9, epsilon_decay=0.99):
        """
        Inicializa o agente com hiperparâmetros ajustáveis.
        - epsilon: controle da exploração.
        - alpha: taxa de aprendizado.
        - gamma: fator de desconto para recompensas futuras.
        - epsilon_decay: reduz a exploração ao longo do tempo.
        """
        self.value_function = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay

    def get_state_key(self, board):
        """
        Converte o tabuleiro em uma tupla para ser usada como chave.
        """
        return tuple(board.flatten())

    def choose_action(self, board, valid_actions):
        """
        Escolhe a próxima jogada usando a estratégia epsilon-greedy.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploração: escolha aleatória
            return random.choice(valid_actions)
        else:
            # Exploração: escolha com base na função de valor
            state_key = self.get_state_key(board)
            best_action = None
            max_value = -float('inf')

            for action in valid_actions:
                next_board = board.copy()
                next_board[action] = 1  # Simula a jogada do agente
                next_state_key = self.get_state_key(next_board)
                value = self.value_function.get(next_state_key, 0)

                if value > max_value:
                    max_value = value
                    best_action = action

            return best_action if best_action else random.choice(valid_actions)

    def update_value_function(self, board, reward, next_board):
        """
        Atualiza a função de valor usando a equação de Bellman.
        """
        state_key = self.get_state_key(board)
        next_state_key = self.get_state_key(next_board)

        old_value = self.value_function.get(state_key, 0)
        next_value = self.value_function.get(next_state_key, 0)

        # Equação de Bellman
        new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
        self.value_function[state_key] = new_value

    def train(self, env, episodes=1000):
        """
        Treina o agente ao longo de vários episódios.
        """
        for episode in range(episodes):
            state = env.reset()
            done = False

            while not done:
                valid_actions = [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]
                action = self.choose_action(state, valid_actions)

                next_state, reward, done = env.step(*action)

                self.update_value_function(state, reward, next_state)

                state = next_state

            # Decai o epsilon para focar mais em estratégias bem-sucedidas ao longo do tempo
            self.epsilon *= self.epsilon_decay

            if episode % 100 == 0:
                print(f"Treinamento - Episódio {episode} concluído. Epsilon: {self.epsilon}")

    def play_against_human(self, env):
        """
        Permite que o humano jogue contra o agente após o treinamento.
        """
        play_again = True
        while play_again:
            state = env.reset()
            env.render()
            done = False

            while not done:
                if env.current_player == 1:  # Turno do agente
                    valid_actions = [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]
                    action = self.choose_action(state, valid_actions)
                    state, _, done = env.step(*action)
                    print("Agente jogou:")
                else:  # Turno do humano
                    while True:
                        try:
                            row, col = map(int, input("Digite sua jogada (linha coluna): ").split())
                            if env.is_valid_action(row, col):
                                state, _, done = env.step(row, col)
                                break
                            else:
                                print("Jogada inválida. Tente novamente.")
                        except ValueError:
                            print("Entrada inválida. Digite dois números entre 0 e 2.")

                env.render()

                if done:
                    reward, _ = env.check_winner()
                    if reward == 1:
                        print("O agente venceu!")
                    elif reward == -1:
                        print("Você venceu!")
                    else:
                        print("Empate!")
            
            # Pergunta se o jogador quer jogar novamente
            play_again = input("Deseja jogar novamente? (s/n): ").lower() == 's'


if __name__ == "__main__":
    # Inicializa o ambiente e o agente
    env = TicTacToeEnv()
    agent = RLAgent()

    # Treina o agente
    agent.train(env, episodes=1000)

    # Jogo contra o humano
    print("\nVamos jogar! Você será o 'O' e o agente será o 'X'.")
    agent.play_against_human(env)
