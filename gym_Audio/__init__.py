from gym.envs.registration import registry, register, make, spec

register(
    id='Audio-v0',
    entry_point='gym_Audio.envs:AudioEnv',
)