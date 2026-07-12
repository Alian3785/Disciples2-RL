# Contributing to Disciples2-RL

Contributors are very welcome!

There are as of right now 4 ways you can contribute to Disciples2-RL:

**Make maps.** Create new maps with different scenarios and challenges for the agent. Use the existing maps in the repository as examples. Preferably, generate new maps with an LLM using a prompt such as: *Create a new map similar to those described in Disciplesmaps, using the following configuration* and then specify the map size, layout, and objects it should contain. Alternatively, you can convert maps created with the original Disciples II editor using the available converter.

**Train agents.** Train agents with new algorithms, parameters, or training run configurations. If your results improve on current SOTA, describe the complete setup in your pull request and include TensorBoard logs from the training run.

**Remove dead code and fix bugs.** A large part of the project was written with the help of coding agents, so there is likely a lot of dead and unnecessary code. Removing it is useful for a project moving forward. You can also submit PR to fix bugs or optimize training performance

**Improve Env accuracy.** The reinforcement learning environment should reproduce the mechanics and behavior of the original Disciples II as accurately as possible. If you know Disciples II well and notice that something works differently from the original game, submit a pull request that corrects the discrepancy. Your knowledge of the original game can be especially valuable to the project!

You may become a contributor after submitting three meaningful (20+ lines) cleanup pull requests. A confirmed bug fix, a performance improvement (>1%), or a fix that brings the environment closer to the behavior of the original Disciples II may qualify you as a contributor immediately after review.

You can request new features by opening an issue, especially if you provide a clear rationale and suggest a possible implementation.
