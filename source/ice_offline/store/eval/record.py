from ice_offline.config.paths import eval_path


class Evaluator:
    def __init__(self, dataset_id: str, agent_id: str, episodes: int = 1) -> None:
        self.episodes = episodes
        self.path = eval_path(dataset_id, agent_id)
        self.history: list[tuple[int, list[float]]] = []

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

        self.history.append((step, returns))
        return sum(returns) / len(returns)

    def save(self) -> None:
        if not self.history:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as f:
            returns_count = len(self.history[0][1])
            header = "step," + ",".join(str(i) for i in range(1, returns_count + 1))
            f.write(f"{header}\n")
            for step, returns in self.history:
                row = ",".join(str(float(value)) for value in returns)
                f.write(f"{step},{row}\n")
