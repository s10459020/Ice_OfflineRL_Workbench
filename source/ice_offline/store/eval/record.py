from ice_offline.config.paths import eval_path


class Evaluator:
    def __init__(self, task_id: str, episodes: int = 1, initialized = False) -> None:
        self.episodes = episodes
        self.path = eval_path(task_id)
        self.initialized = initialized

    def eval(self, step: int, agent, eval_env) -> float:
        returns: list[float] = []
        for _ in range(self.episodes):
            obs, _ = eval_env.reset()
            done = False
            total_return = 0.0

            while not done:
                action = agent.act_best(obs)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = bool(terminated or truncated)
                total_return += float(reward)

            returns.append(total_return)

        self.append(step, returns)
        return sum(returns) / len(returns)

    def append(self, step: int, returns: list[float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.initialized else "w"

        with self.path.open(mode, encoding="utf-8", newline="") as f:
            if not self.initialized:
                header = "step," + ",".join(str(i) for i in range(1, len(returns) + 1))
                f.write(f"{header}\n")
                self.initialized = True
            row = ",".join(str(float(value)) for value in returns)
            f.write(f"{step},{row}\n")
