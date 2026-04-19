import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteBCAgent:
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        learning_rate: float = 1e-3,
        beta: float = 0.5,
        hidden_units: tuple[int, ...] = (256, 256),
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.beta = beta
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # d3rlpy discrete BC default encoder: MLP [256, 256] + ReLU
        self.encoder = self._build_mlp(
            input_dim=self.obs_dim,
            hidden_units=hidden_units,
        ).to(self.device)
        last_dim = hidden_units[-1] if len(hidden_units) > 0 else self.obs_dim
        self.policy_head = nn.Linear(last_dim, self.n_actions).to(self.device)

        self.optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy_head.parameters()),
            lr=learning_rate,
        )

    # ====================
    # Public API
    # ====================
    def act(self, observation: np.ndarray) -> int:
        # greedy action (same as d3rlpy predict_best_action)
        self.encoder.eval()
        self.policy_head.eval()
        with torch.no_grad():
            obs_t = self._to_obs_tensor(observation)
            logits = self._policy_logits(obs_t)
            return int(logits.argmax(dim=1).item())

    def sample(self, observation: np.ndarray) -> int:
        # stochastic action sampled from categorical policy
        self.encoder.eval()
        self.policy_head.eval()
        with torch.no_grad():
            obs_t = self._to_obs_tensor(observation)
            logits = self._policy_logits(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())

    def update(self, o: np.ndarray, a: int, r: float, o_: np.ndarray, done: bool) -> dict[str, float]:
        # BC is supervised imitation; r/o_/done are ignored for interface compatibility
        del r, o_, done
        obs_batch = np.asarray(o, dtype=np.float32).reshape(1, -1)
        act_batch = np.asarray([a], dtype=np.int64)
        return self.update_batch(obs_batch, act_batch)

    def update_batch(self, observations: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor) -> dict[str, float]:
        self.encoder.train()
        self.policy_head.train()

        obs_t = self._to_obs_tensor(observations)
        act_t = self._to_action_tensor(actions)

        self.optim.zero_grad()
        total_loss, imitation_loss, regularization_loss = self._loss(obs_t, act_t)
        total_loss.backward()
        self.optim.step()

        return {
            "loss": float(total_loss.item()),
            "imitation_loss": float(imitation_loss.item()),
            "regularization_loss": float(regularization_loss.item()),
        }

    def predict_proba(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        self.encoder.eval()
        self.policy_head.eval()
        with torch.no_grad():
            obs_t = self._to_obs_tensor(observations)
            logits = self._policy_logits(obs_t)
            probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    # ====================
    # actor mathmatics
    # ====================
    def _policy_logits(self, obs_t: torch.Tensor) -> torch.Tensor:
        # logits = policy_head(encoder(obs))
        latent = self.encoder(obs_t)
        return self.policy_head(latent)

    def _loss(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # d3rlpy discrete BC:
        # L = NLL(log_softmax(logits), action) + beta * mean(logits^2)
        logits = self._policy_logits(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        # Match d3rlpy: penalty is computed on dist.logits (normalized logits).
        penalty = (dist.logits ** 2).mean()
        log_probs = F.log_softmax(logits, dim=1)
        imitation_loss = F.nll_loss(log_probs, act_t.view(-1))
        regularization_loss = self.beta * penalty
        total_loss = imitation_loss + regularization_loss
        return total_loss, imitation_loss, regularization_loss

    # ====================
    # utils
    # ====================
    def _build_mlp(self, input_dim: int, hidden_units: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden in hidden_units:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        return nn.Sequential(*layers)

    def _to_obs_tensor(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(observation, torch.Tensor):
            obs_t = observation.to(self.device, dtype=torch.float32)
        else:
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

    def _to_action_tensor(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            act_t = action.to(self.device, dtype=torch.long)
        else:
            act_t = torch.as_tensor(action, dtype=torch.long, device=self.device)
        if act_t.ndim == 0:
            act_t = act_t.unsqueeze(0)
        return act_t
