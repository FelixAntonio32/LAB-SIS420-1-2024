#INTEGRANTES:
#Fernandez Muruchi Lisbeth
#Flores Yampara Felix Antonio

#Selección de acciones con intervalo de confianza
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    env = gym.make('FrozenLake-v1')
    
    # Inicializa la tabla Q con ceros
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    
    # Inicializa la tabla N con unos (para evitar división por cero)
    n_table = np.ones([env.observation_space.n, env.action_space.n])
    
    learning_rate = 0.1
    discount_factor = 0.95
    c = 1.0  # Coeficiente de exploración
    rng = np.random.default_rng()
    
    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_per_episode = np.zeros(episodes)
    total_steps = 0
    
    for i in range(episodes):
        # Reinicia el entorno cada 10000 episodios alternando entre modos con y sin renderizado
        if (i * 1) % 10000 == 0:
            env.close()
            env = gym.make('FrozenLake-v1', render_mode='human')
            print(f'Episodio {i + 1}')
        else:
            env.reset()
            env = gym.make('FrozenLake-v1')
            
        # Reinicia el entorno y establece el estado inicial
        state = env.reset()[0]
        done = False
        
        while not done:
            total_steps += 1
            
            # Selección de acción con UCB
            ucb_values = q_table[state, :] + c * np.sqrt(np.log(total_steps) / n_table[state, :])
            action = np.argmax(ucb_values)
            
            # Realiza la acción seleccionada y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Actualiza la tabla Q con la nueva información obtenida
            old_value = q_table[state, action]
            next_max = np.max(q_table[new_state, :])
            q_table[state, action] = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            
            # Actualiza la tabla N
            n_table[state, action] += 1
            
            # Actualiza el estado actual para el siguiente paso
            state = new_state
            
            if terminated or truncated:
                done = True
        
        # Registra si el agente obtuvo una recompensa (llegó al objetivo) en este episodio
        rewards_per_episode[i] = reward
        
        # Imprime el progreso cada 10000 episodios
        if (i + 1) % 10000 == 0:
            print(f'Episodio {i + 1} recompensa: {reward}')
        
    # Cierra el entorno al finalizar el entrenamiento
    env.close()
    
    # Imprime la mejor tabla Q obtenida
    print('Tabla Q final:')
    print(q_table)
    
    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Suma de recompensas acumuladas por episodios')
    plt.show()
    
if __name__ == '__main__':
    train(100000)
