"""A compact yet complete vanilla RNN implementation for learning purposes.

This module focuses on the core mechanics of a recurrent neural network:

1. Embedding lookup from token ids to vectors
2. Recurrent hidden state transition
3. Output projection to vocabulary logits
4. Softmax cross-entropy loss
5. Backpropagation through time (BPTT)
6. SGD update with optional gradient clipping

The code intentionally avoids deep framework abstractions so the math and
tensor shapes remain visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


Array = np.ndarray


def softmax(logits: Array) -> Array:
    """Numerically stable softmax for 1D or 2D arrays."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def cross_entropy_from_logits(logits: Array, targets: Array) -> float:
    """Average cross-entropy loss.

    Args:
        logits: Shape (batch_size, vocab_size)
        targets: Shape (batch_size,)
    """
    probs = softmax(logits)
    batch_size = targets.shape[0]
    safe_probs = np.clip(probs[np.arange(batch_size), targets], 1e-12, 1.0)
    return float(-np.mean(np.log(safe_probs)))


@dataclass
class StepCache:
    """Saved values for one time step, used during BPTT."""

    token_ids: Array
    x_embed: Array
    h_prev: Array
    h_t: Array
    logits: Array


class SimpleRNN:
    """Educational vanilla RNN implemented with NumPy.

    Shapes:
        token_ids: (batch_size, seq_len)
        embeddings: (vocab_size, embed_dim)
        hidden state: (batch_size, hidden_dim)
        logits per step: (batch_size, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if vocab_size <= 0 or embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("vocab_size, embed_dim, hidden_dim must be positive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or vocab_size

        rng = np.random.default_rng(seed)
        scale = 0.1

        # Embedding table: maps token id -> dense vector.
        self.embedding = rng.normal(0.0, scale, size=(vocab_size, embed_dim))

        # Recurrent core:
        # h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
        self.W_xh = rng.normal(0.0, scale, size=(embed_dim, hidden_dim))
        self.W_hh = rng.normal(0.0, scale, size=(hidden_dim, hidden_dim))
        self.b_h = np.zeros((1, hidden_dim))

        # Output layer:
        # logits_t = h_t @ W_hy + b_y
        self.W_hy = rng.normal(0.0, scale, size=(hidden_dim, self.output_dim))
        self.b_y = np.zeros((1, self.output_dim))

        self.grads: Dict[str, Array] = self._zero_grads()

    def _zero_grads(self) -> Dict[str, Array]:
        return {
            "embedding": np.zeros_like(self.embedding),
            "W_xh": np.zeros_like(self.W_xh),
            "W_hh": np.zeros_like(self.W_hh),
            "b_h": np.zeros_like(self.b_h),
            "W_hy": np.zeros_like(self.W_hy),
            "b_y": np.zeros_like(self.b_y),
        }

    def zero_grad(self) -> None:
        """Reset accumulated gradients."""
        self.grads = self._zero_grads()

    def init_hidden(self, batch_size: int) -> Array:
        return np.zeros((batch_size, self.hidden_dim))

    def step_forward(self, token_ids: Array, h_prev: Array) -> Tuple[Array, Array, StepCache]:
        """Forward pass for a single time step."""
        x_embed = self.embedding[token_ids]
        h_linear = x_embed @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h_t = np.tanh(h_linear)
        logits = h_t @ self.W_hy + self.b_y

        cache = StepCache(
            token_ids=token_ids.copy(),
            x_embed=x_embed,
            h_prev=h_prev,
            h_t=h_t,
            logits=logits,
        )
        return h_t, logits, cache

    def forward(self, token_ids: Array, h0: Optional[Array] = None) -> Tuple[Array, Array, List[StepCache]]:
        """Forward pass across the whole sequence.

        Args:
            token_ids: Shape (batch_size, seq_len)
            h0: Optional initial hidden state of shape (batch_size, hidden_dim)
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, seq_len)")

        batch_size, seq_len = token_ids.shape
        h_t = self.init_hidden(batch_size) if h0 is None else h0.copy()

        logits_per_step = []
        hidden_states = []
        caches: List[StepCache] = []

        for t in range(seq_len):
            h_t, logits_t, cache_t = self.step_forward(token_ids[:, t], h_t)
            hidden_states.append(h_t)
            logits_per_step.append(logits_t)
            caches.append(cache_t)

        return np.stack(logits_per_step, axis=1), np.stack(hidden_states, axis=1), caches

    def compute_loss(self, logits: Array, targets: Array) -> float:
        """Sequence-level average cross-entropy loss.

        Args:
            logits: Shape (batch_size, seq_len, vocab_size)
            targets: Shape (batch_size, seq_len)
        """
        if logits.ndim != 3 or targets.ndim != 2:
            raise ValueError("Invalid shapes for logits or targets")

        batch_size, seq_len, vocab_size = logits.shape
        flat_logits = logits.reshape(batch_size * seq_len, vocab_size)
        flat_targets = targets.reshape(batch_size * seq_len)
        return cross_entropy_from_logits(flat_logits, flat_targets)

    def backward(self, targets: Array, caches: List[StepCache]) -> float:
        """Backpropagation through time.

        Args:
            targets: Shape (batch_size, seq_len)
            caches: Step caches collected during forward()

        Returns:
            Average cross-entropy loss across all time steps.
        """
        if targets.ndim != 2:
            raise ValueError("targets must have shape (batch_size, seq_len)")
        if len(caches) != targets.shape[1]:
            raise ValueError("Number of caches must equal sequence length")

        self.zero_grad()
        batch_size, seq_len = targets.shape
        dh_next = np.zeros((batch_size, self.hidden_dim))
        total_loss = 0.0

        for t in reversed(range(seq_len)):
            cache = caches[t]
            logits = cache.logits
            probs = softmax(logits)

            # dL/dlogits for average CE loss
            dlogits = probs
            dlogits[np.arange(batch_size), targets[:, t]] -= 1.0
            dlogits /= batch_size * seq_len

            total_loss += cross_entropy_from_logits(logits, targets[:, t]) / seq_len

            self.grads["W_hy"] += cache.h_t.T @ dlogits
            self.grads["b_y"] += np.sum(dlogits, axis=0, keepdims=True)

            dh = dlogits @ self.W_hy.T + dh_next

            # tanh' = 1 - tanh(x)^2
            dh_raw = dh * (1.0 - cache.h_t ** 2)

            self.grads["W_xh"] += cache.x_embed.T @ dh_raw
            self.grads["W_hh"] += cache.h_prev.T @ dh_raw
            self.grads["b_h"] += np.sum(dh_raw, axis=0, keepdims=True)

            dx_embed = dh_raw @ self.W_xh.T
            dh_next = dh_raw @ self.W_hh.T

            # Embedding is a lookup table, so gradients are accumulated by token id.
            np.add.at(self.grads["embedding"], cache.token_ids, dx_embed)

        return float(total_loss)

    def clip_gradients(self, max_norm: float = 5.0) -> float:
        """Clip gradients by global norm to improve training stability."""
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")

        total_sq_norm = 0.0
        for grad in self.grads.values():
            total_sq_norm += float(np.sum(grad ** 2))

        total_norm = np.sqrt(total_sq_norm)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-12)
            for name in self.grads:
                self.grads[name] *= scale
        return float(total_norm)

    def step(self, learning_rate: float = 1e-2) -> None:
        """SGD parameter update."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.embedding -= learning_rate * self.grads["embedding"]
        self.W_xh -= learning_rate * self.grads["W_xh"]
        self.W_hh -= learning_rate * self.grads["W_hh"]
        self.b_h -= learning_rate * self.grads["b_h"]
        self.W_hy -= learning_rate * self.grads["W_hy"]
        self.b_y -= learning_rate * self.grads["b_y"]

    def train_batch(
        self,
        inputs: Array,
        targets: Array,
        learning_rate: float = 1e-2,
        clip_norm: Optional[float] = 5.0,
        h0: Optional[Array] = None,
    ) -> Dict[str, float]:
        """Run one full training step on a batch."""
        logits, _, caches = self.forward(inputs, h0=h0)
        loss = self.backward(targets, caches)

        grad_norm = 0.0
        if clip_norm is not None:
            grad_norm = self.clip_gradients(clip_norm)

        self.step(learning_rate)

        predictions = np.argmax(logits, axis=-1)
        accuracy = float(np.mean(predictions == targets))

        return {
            "loss": loss,
            "accuracy": accuracy,
            "grad_norm": grad_norm,
        }

    def predict(self, inputs: Array, h0: Optional[Array] = None) -> Tuple[Array, Array]:
        """Return token predictions and final hidden states."""
        logits, hidden_states, _ = self.forward(inputs, h0=h0)
        return np.argmax(logits, axis=-1), hidden_states[:, -1, :]

    def generate(
        self,
        start_token: int,
        max_length: int,
        temperature: float = 1.0,
        h0: Optional[Array] = None,
        stochastic: bool = True,
    ) -> List[int]:
        """Generate a sequence token by token."""
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if not 0 <= start_token < self.vocab_size:
            raise ValueError("start_token is out of vocabulary range")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        h_t = self.init_hidden(1) if h0 is None else h0.copy()
        current_token = np.array([start_token], dtype=np.int64)
        generated = [start_token]

        for _ in range(max_length - 1):
            h_t, logits, _ = self.step_forward(current_token, h_t)
            scaled_logits = logits[0] / temperature
            probs = softmax(scaled_logits)

            if stochastic:
                next_token = int(np.random.choice(self.output_dim, p=probs))
            else:
                next_token = int(np.argmax(probs))

            generated.append(next_token)
            current_token = np.array([next_token], dtype=np.int64)

        return generated


def create_language_modeling_data(sequence: List[int]) -> Tuple[Array, Array]:
    """Convert a token sequence into next-token prediction data.

    Example:
        sequence = [1, 5, 3, 8]
        inputs  = [[1, 5, 3]]
        targets = [[5, 3, 8]]
    """
    if len(sequence) < 2:
        raise ValueError("sequence length must be at least 2")

    inputs = np.array(sequence[:-1], dtype=np.int64)[None, :]
    targets = np.array(sequence[1:], dtype=np.int64)[None, :]
    return inputs, targets


if __name__ == "__main__":
    np.random.seed(42)

    # Tiny toy example: learn a repeating token pattern.
    toy_sequence = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    inputs, targets = create_language_modeling_data(toy_sequence)

    model = SimpleRNN(
        vocab_size=4,
        embed_dim=8,
        hidden_dim=16,
        seed=42,
    )

    for epoch in range(1, 301):
        metrics = model.train_batch(
            inputs=inputs,
            targets=targets,
            learning_rate=0.1,
            clip_norm=5.0,
        )
        if epoch % 50 == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={metrics['loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} "
                f"grad_norm={metrics['grad_norm']:.4f}"
            )

    generated = model.generate(start_token=0, max_length=8, temperature=0.8, stochastic=False)
    print("generated:", generated)
