from pettingzoo.mpe import simple_speaker_listener_v4


env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
_, _ = env.reset()
for agent in env.agents:
    print(f'agent observation space {env.observation_space(agent)}')
obs, info = env.reset()
print(f'initial observation: {obs} debug info: {info}')
terminal = [False] * env.max_num_agents
while not any(terminal):
    actions = {}
    for agent in env.agents:
        actions[agent] = env.action_space(agent).sample()
    obs_, reward, done, trunc, info = env.step(actions)
    terminal = [d or t for d, t in zip(done.values(), trunc.values())]
print(f'actions taken {actions}')
print(f'obs values {obs.values()}')
obs = list(obs.values())
print(f'obs as a list {obs}')
