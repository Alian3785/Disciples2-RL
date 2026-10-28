# Contributing to Disciples2-RL

Contributors are very welcome!

There are as of right now 4 ways you can contribute to Disciples2-RL:

**Make maps.** Create new maps with different scenarios and challenges for the agent. Use the existing maps in the repository as examples. Preferably, generate new maps with an LLM using a prompt such as: *Create a new map similar to those described in Disciplesmaps, using the following configuration* and then specify the map size, layout, and objects it should contain. Alternatively, you can convert maps created with the original Disciples II editor using the available converter.

**Train agents.** Train agents with new algorithms, parameters, or training run configurations. If your results improve on current SOTA, describe the complete setup in your pull request and include TensorBoard logs from the training run.

**Remove dead code and fix bugs.** A large part of the project was written with the help of coding agents, so there is likely a lot of dead and unnecessary code. Removing it is useful for a project moving forward. You can also submit PR to fix bugs or optimize training performance

**Improve Env accuracy.** The reinforcement learning environment should reproduce the mechanics and behavior of the original Disciples II as accurately as possible. If you know Disciples II well and notice that something works differently from the original game, submit a pull request that corrects the discrepancy. Your knowledge of the original game can be especially valuable to the project!

You may become a contributor after submitting three meaningful (20+ lines) cleanup pull requests. A confirmed bug fix, a performance improvement (>1%), or a fix that brings the environment closer to the behavior of the original Disciples II may qualify you as a contributor immediately after review.

You can request new features by opening an issue, especially if you provide a clear rationale and suggest a possible implementation.

## Работа в локальном проекте

Правила проверки, источников механик и запуска GPU-задач находятся в [AGENTS.md](AGENTS.md). Команды — в [руководстве по обучению](docs/TRAINING.md#testing). Описания правил поддерживаются в тематических справочниках, результаты отдельных запусков — в `results/`.

## Правила участия в исходном Stoix

Следующие требования относятся к отправке изменений в исходный проект Stoix.

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

### Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

### Installing Stoix for development

To develop features for Stoix, clone the repository and install the core and development dependencies as follows:

```bash
git clone https://github.com/EdanToledo/Stoix.git
cd Stoix
pip install -e .[dev]
```

### Installing Pre-Commit Hooks

Install the pre-commit hooks:
```bash
pre-commit install
pre-commit install -t commit-msg
```
You can run all the pre-commit hooks on all files as follows:
```bash
pre-commit run --all-files
```

### Naming Conventions
#### Branch Names
We name our feature and bugfix branches as follows - `feature/[BRANCH-NAME]`, `bugfix/[BRANCH-NAME]` or `maintenance/[BRANCH-NAME]`. Please ensure `[BRANCH-NAME]` is hyphen delimited.
#### Commit Messages
We follow the conventional commits [standard](https://www.conventionalcommits.org/en/v1.0.0/).

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
