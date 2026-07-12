So, this is the project to train a Superhuman bot for Disciples 2, popular turn-based strategy game. There are 3 stages:

1. **Single-player scenarios**
   First, an agent is trained to complete individual single-player scenarios. Autoresearch is used to test multiple configurations.
   [Preliminary roadmap](https://github.com/users/Alian3785/projects/2)

2. **Randomly generated and unseen maps**
   Next, a more general agent is trained to play previously unseen maps, face randomly generated enemies, and handle a wider variety of situations.

3. **Self-play training**
   Finally, an AlphaZero-like agent is trained through self-play, with the goal of competing against the best *Disciples II* players and eventually beating the champion—whoever that may be.
   [Why not just use AlphaZero right away?](https://github.com/Alian3785/Disciples2-RL/blob/main/Why%20not%20just%20use%20AlphaZero.md)

All features of the original game have been implemented in Python, except for rod planters, the recruitment of new heroes, the hiring of thieves, non-interactive map objects, сastle guard and the original visuals.

You can run agent training for a real Disciples 2 scenario "A Return To Simpler Times" (with the exceptions above) with the following command. By default, a slightly modified Maskable PPO algorithm from [Stable Baselines 3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) is used.

```bash
python Big_map/train_campaign.py
```

The agent is being trained for *Disciples II: Rise of the Elves*:
[Disciples II: Rise of the Elves on Steam](https://store.steampowered.com/app/1630/Disciples_II_Rise_of_the_Elves/)



