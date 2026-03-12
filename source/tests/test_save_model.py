from pathlib import Path

from tests.test_Q_table import train_q_learning


REPO_ROOT = Path(__file__).resolve().parents[2]


def train_and_save_model(
    model_name: str = "q_table_agent.pkl",
    env_id: str = "MiniGrid-Empty-5x5-v0",
) -> Path:
    agent = train_q_learning(env_id=env_id)
    model_path = REPO_ROOT / "model" / model_name
    saved_path = agent.save(model_path)
    print(f"saved model: {saved_path} | q_states={len(agent.q_table)}")
    return saved_path


if __name__ == "__main__":
    train_and_save_model()
