"""Fixed-aggressiveness speculator (validation opponent).

:class:`FixedChiAgent` trades ``χ · (v_t − v̄)`` every period, snapped to the
nearest quantity of the action table at the current value index, and never
learns. It is the opponent of the two validation runs: against an opponent who plays the Nash
aggressiveness and never reacts, a correct learner should end near its best
response, which is the Nash level itself.

Config key ``chi``: a float, or the strings ``"nash"`` / ``"cartel"``, which
take ``chi_N`` / ``chi_M`` from the config (the runner passes the closed-form
benchmarks of the session).

:meth:`greedy_policy` returns the fixed index per state so that the runner's
convergence check and stability diagnostic work in a mixed population.
``visit_count`` is ``None`` (nothing is tracked).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_trading.agents.base import Agent, Transition


class FixedChiAgent(Agent):
    """Plays the grid action nearest to ``χ · (v_t − v̄)``; never learns."""

    def __init__(
        self,
        agent_id: int,
        config: dict[str, Any],
        grids: dict[str, np.ndarray],
    ) -> None:
        """Resolve ``χ`` and precompute the action index per value index.

        Args:
            agent_id: Integer identifier (0-indexed).
            config: Needs ``n_v``, ``n_x``, ``n_p`` and ``chi`` (float or
                ``"nash"`` / ``"cartel"``; the strings need ``chi_N`` /
                ``chi_M``). Optional ``v_bar`` (default 0.0).
            grids: Needs ``v_grid`` and ``action_table`` (shape ``(n_v, n_x)``).

        Raises:
            ValueError: On an unknown ``chi`` string or a mis-shaped table.
        """
        n_v = int(config["n_v"])
        n_x = int(config["n_x"])
        n_p = int(config["n_p"])
        super().__init__(agent_id=agent_id, n_actions=n_x)
        self.n_p, self.n_v, self.n_x = n_p, n_v, n_x
        self.v_bar: float = float(config.get("v_bar", 0.0))

        chi_cfg = config["chi"]
        if isinstance(chi_cfg, str):
            key = chi_cfg.strip().lower()
            if key == "nash":
                self.chi = float(config["chi_N"])
            elif key == "cartel":
                self.chi = float(config["chi_M"])
            else:
                raise ValueError(
                    f"chi must be a number, 'nash' or 'cartel', got {chi_cfg!r}."
                )
        else:
            self.chi = float(chi_cfg)
        self.chi_label: str = str(chi_cfg)

        v_grid = np.asarray(grids["v_grid"], dtype=np.float64)
        table = np.asarray(grids["action_table"], dtype=np.float64)
        if table.shape != (n_v, n_x):
            raise ValueError(
                f"action_table has shape {table.shape}, expected ({n_v}, {n_x})."
            )
        target = self.chi * (v_grid - self.v_bar)  # (n_v,)
        self.action_by_v: np.ndarray = np.abs(table - target[:, None]).argmin(axis=1)

    def act(self, state: np.ndarray) -> int:
        """The precomputed index for the current value bin ``state[2]``."""
        return int(self.action_by_v[int(state[2])])

    def update(self, transition: Transition) -> None:
        """No-op: the policy is fixed."""

    def reset(self) -> None:
        """No-op."""

    def greedy_policy(self) -> np.ndarray:
        """The fixed index per state, shape ``(n_p, n_v, n_v)``."""
        return np.broadcast_to(
            self.action_by_v[None, None, :], (self.n_p, self.n_v, self.n_v)
        ).copy()
