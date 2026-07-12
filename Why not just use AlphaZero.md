# Why Not Just Use AlphaZero?

At first glance, it seems simple: AlphaZero learned to play Go and chess better than anyone by playing millions of games against itself and learning from them. So we should be able to make it play millions of games of Disciples II, train in the same way, and end up with an unbeatable bot. Unfortunately, it is not that simple. Let us describe the problem in plain language, adding the relevant reinforcement learning term at the end of each paragraph.

Vanilla AlphaZero is not designed to handle games with fog of war, which Disciples II has. The fog hides parts of the map and the opponent's actions; nothing like that exists in Go. **Partial observability.**

Units in Disciples II can miss, and they may act in a different order depending on randomness. Nothing similar happens in chess, where the same move in the same position always produces the same result. **Stochasticity.**

Finally, Disciples II is an asymmetric game: players may control different factions and encounter different opponents. This does not happen in simple board games. **Asymmetry.**

And these are not the only differences. The number of available actions grows rapidly over the course of a game, from giving different elixirs to different units to hiring city guards, constructing buildings in the capital, equipping items, and much more. **Combinatorial explosion.**

All of this makes vanilla AlphaZero unsuitable for Disciples II.

Almost all of these features, from partial observability to asymmetry, were addressed by a later algorithm, AlphaStar. However, its foundation includes imitation learning from game replays. I do not have any Disciples II replays. In fact, no such replay dataset exists. Nor do I have the tens of millions of dollars that were spent on the computing resources required to train AlphaStar.
