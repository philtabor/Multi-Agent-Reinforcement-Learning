import numpy as np
from agent import Agent
from pettingzoo.mpe import simple_speaker_listener_v4


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    env = simple_speaker_listener_v4.parallel_env(
            continuous_actions=True)
    scenario = 'simple_speaker_listener'

    initial_temp = env.reset()
    n_agents = env.max_num_agents

    agents = []

    for agent in env.agents:
        input_dims = env.observation_space(agent).shape[0]
        n_actions = env.action_space(agent).shape[0]

        agents.append(Agent(alpha=1e-3, beta=1e-3,
                      input_dims=input_dims, tau=0.01, gamma=0.95,
                      batch_size=1024, fc1_dims=64, fc2_dims=64,
                      n_actions=n_actions))

    N_GAMES = 25_000
    PRINT_INTERVAL = 500
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        for agent in agents:
            agent.load_checkpoint()

    total_steps = 0

    for i in range(N_GAMES):
        observation, _ = env.reset()
        terminal = [False] * n_agents
        score = 0
        observation = list(observation.values())

        while not any(terminal):
            action = [agent.choose_action(observation[idx])
                      for idx, agent in enumerate(agents)]
            action = {agent: act for agent, act in zip(env.agents, action)}
            observation_, reward, done, trunc, info = env.step(action)

            observation_ = list(observation_.values())
            reward = list(reward.values())
            done = list(done.values())
            trunc = list(trunc.values())
            action = list(action.values())

            terminal = [d or t for d, t in zip(done, trunc)]

            for idx, agent in enumerate(agents):
                agent.remember(observation[idx], action[idx],
                               reward[idx], observation_[idx], terminal[idx])
            if total_steps % 100 == 0 and not evaluate:
                for agent in agents:
                    agent.learn()
            score += sum(reward)
            observation = observation_
            total_steps += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f'episode {i} avg score {avg_score:.1f}')
