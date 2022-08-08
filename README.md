# InterpretableModel

文章 *Interpretable Deep Reinforcement Learning via Imitation Learning* 的开源代码

## ENV

- 'CartPole-v0' 
- 'LunarLander-v2'

## ALGORITHM

- ppo 
- dqn



## Command

### PPO

####  'CartPole-v0' 

 **100 个 episode 内求解** 

```
# train ppo
python train_rl.py --env CartPole-v0 ppo --use-recurrent-layer --save-model
# test ppo
python test_rl.py --env CartPole-v0 --save-time 01_22_11_02 ppo --use-recurrent-layer
# collect data
python collect_data.py --env CartPole-v0 --target-label action --save-time 01_22_11_02 ppo --use-recurrent-layer
```

#### 'LunarLander-v2'

**300 个 episode 内求解**

```
# train ppo
python train_rl.py --env LunarLander-v2 ppo --use-recurrent-layer --save-model
# test ppo
python test_rl.py --env LunarLander-v2 --save-time 01_21_21_12 ppo --use-recurrent-layer
# collect data
python collect_data.py --env LunarLander-v2 --target-label action --save-time 01_21_21_12 ppo --use-recurrent-layer
```



### DQN

####  'CartPole-v0'

 **100 个 episode 内求解** 

```
# train dqn 
python train_rl.py --env CartPole-v0 dqn --save-model --solved-reward 195
# test ppo
python test_rl.py --env CartPole-v0 --save-time 02_12_17_10 dqn
# collect data
python collect_data.py --env CartPole-v0 --target-label value --save-time 02_12_17_10 ppo --use-recurrent-layer
```
