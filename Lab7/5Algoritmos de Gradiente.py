#INTEGRANTES:
#Fernandez Muruchi Lisbeth
#Flores Yampara Felix Antonio

#Algoritmos de Gradiente
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    env = gym.make('FrozenLake-v1')
    
    # Número de acciones y estados
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    
    # Inicializa las preferencias de acción con ceros
    preferences = np.zeros([n_states, n_actions])
    
    learning_rate = 0.1
    discount_factor = 0.95
    baseline = 0.0  # Baseline inicial para la actualización de preferencias
    rng = np.random.default_rng()
    
    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_per_episode = np.zeros(episodes)
    
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
        
        episode_rewards = []
        episode_actions = []
        
        while not done:
            # Calcula la política softmax de las preferencias de acción
            action_probs = np.exp(preferences[state, :]) / np.sum(np.exp(preferences[state, :]))
            
            # Selecciona una acción basada en la política softmax
            action = rng.choice(n_actions, p=action_probs)
            
            # Realiza la acción seleccionada y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action)
            
            if terminated or truncated:
                done = True
        
        # Calcula las recompensas acumuladas descontadas
        cumulative_rewards = np.zeros_like(episode_rewards, dtype=np.float64)
        cumulative_reward = 0.0
        for t in reversed(range(len(episode_rewards))):
            cumulative_reward = episode_rewards[t] + discount_factor * cumulative_reward
            cumulative_rewards[t] = cumulative_reward
        
        # Actualiza las preferencias de acción con algoritmo de gradiente
        for t in range(len(episode_rewards)):
            state = env.reset()[0]
            action = episode_actions[t]
            
            # Calcula el gradiente logarítmico para la acción seleccionada
            action_prob = action_probs[action]
            grad_log = 1.0 - action_prob
            
            preferences[state, action] += learning_rate * (cumulative_rewards[t] - baseline) * grad_log
        
        # Registra la recompensa total del episodio
        rewards_per_episode[i] = np.sum(episode_rewards)
        
        # Imprime el progreso cada 10000 episodios
        if (i + 1) % 10000 == 0:
            print(f'Episodio {i + 1} recompensa: {np.sum(episode_rewards)}')
        
    # Cierra el entorno al finalizar el entrenamiento
    env.close()
    
    # Imprime las preferencias de acción finales
    print('Preferencias de acción finales:')
    print(preferences)
    
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
