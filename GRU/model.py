"""Manual GRU implementation with PyTorch tensors and autograd.

This file does not use ``nn.GRU``. It builds the GRU recurrence from:

1. ``nn.Embedding`` for token lookup
2. ``nn.Parameter`` for gate weights
3. explicit time-step recurrence in Python
4. autograd for backward propagation

Compared with LSTM, GRU removes the separate cell state and uses fewer gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class Batch:
    inputs: Tensor
    targets: Tensor


class GRUCell(nn.Module):
    """A single GRU cell implemented from gate equations."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Update gate z_t: controls how much old state is kept.
        self.W_z = nn.Parameter(torch.empty(input_dim + hidden_dim, hidden_dim))
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))

        # Reset gate r_t
        self.W_r = nn.Parameter(torch.empty(input_dim + hidden_dim, hidden_dim))
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))

        # Candidate hidden state n_t (sometimes written as h_t_tilde)
        self.W_n = nn.Parameter(torch.empty(input_dim + hidden_dim, hidden_dim))
        self.b_n = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("W_"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """One recurrent step.

        Args:
            x_t: Shape (batch_size, input_dim)
            h_prev: Shape (batch_size, hidden_dim)
        """
        combined = torch.cat([x_t, h_prev], dim=-1)

        z_t = torch.sigmoid(combined @ self.W_z + self.b_z)
        r_t = torch.sigmoid(combined @ self.W_r + self.b_r)

        # Reset gate controls how much old state is used when building candidate state.
        candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
        n_t = torch.tanh(candidate_input @ self.W_n + self.b_n)

        # Match the common GRU diagram:
        # z_t keeps old state, (1 - z_t) writes new candidate state.
        h_t = z_t * h_prev + (1.0 - z_t) * n_t
        return h_t


class GRULanguageModel(nn.Module):
    """Language model built from a handwritten GRU cell."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("vocab_size, embed_dim and hidden_dim must be positive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = GRUCell(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Tensor:
        if device is None:
            device = self.embedding.weight.device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        token_ids: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Manual unrolling over time.

        Args:
            token_ids: Shape (batch_size, seq_len)
            hidden: Shape (batch_size, hidden_dim)
        """
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape (batch_size, seq_len)")

        batch_size, seq_len = token_ids.shape
        if hidden is None:
            h_t = self.init_hidden(batch_size, token_ids.device)
        else:
            h_t = hidden

        embeddings = self.embedding(token_ids)
        outputs = []

        for t in range(seq_len):
            x_t = embeddings[:, t, :]
            h_t = self.cell(x_t, h_t)
            logits_t = self.output(h_t)
            outputs.append(logits_t)

        logits = torch.stack(outputs, dim=1)
        return logits, h_t

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.dim() != 3 or targets.dim() != 2:
            raise ValueError("logits must be (batch_size, seq_len, vocab_size) and targets (batch_size, seq_len)")

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)
        return nn.functional.cross_entropy(logits_flat, targets_flat)

    @torch.no_grad()
    def predict(self, token_ids: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        self.eval()
        logits, hidden_n = self.forward(token_ids, hidden)
        predictions = torch.argmax(logits, dim=-1)
        return predictions, hidden_n

    @torch.no_grad()
    def generate(
        self,
        start_token: int,
        max_length: int,
        temperature: float = 1.0,
        stochastic: bool = True,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.eval()
        if device is None:
            device = self.embedding.weight.device

        current = torch.tensor([[start_token]], dtype=torch.long, device=device)
        hidden = self.init_hidden(batch_size=1, device=device)
        generated = [start_token]

        for _ in range(max_length - 1):
            logits, hidden = self.forward(current, hidden)
            step_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(step_logits, dim=-1)

            if stochastic:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            generated.append(token_id)
            current = next_token

        return generated


def create_language_modeling_batch(sequence: List[int], device: Optional[torch.device] = None) -> Batch:
    if len(sequence) < 2:
        raise ValueError("sequence length must be at least 2")

    inputs = torch.tensor(sequence[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(sequence[1:], dtype=torch.long, device=device).unsqueeze(0)
    return Batch(inputs=inputs, targets=targets)


def train_one_epoch(
    model: GRULanguageModel,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    clip_norm: Optional[float] = 5.0,
) -> dict:
    model.train()
    optimizer.zero_grad()

    logits, _ = model(batch.inputs)
    loss = model.compute_loss(logits, batch.targets)
    loss.backward()

    grad_norm = None
    if clip_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm))

    optimizer.step()

    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        accuracy = float((predictions == batch.targets).float().mean().item())

    return {
        "loss": float(loss.item()),
        "accuracy": accuracy,
        "grad_norm": 0.0 if grad_norm is None else grad_norm,
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toy_sequence = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    batch = create_language_modeling_batch(toy_sequence, device=device)

    model = GRULanguageModel(
        vocab_size=4,
        embed_dim=8,
        hidden_dim=16,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(1, 201):
        metrics = train_one_epoch(model, batch, optimizer, clip_norm=5.0)
        if epoch % 40 == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={metrics['loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} "
                f"grad_norm={metrics['grad_norm']:.4f}"
            )

    generated = model.generate(start_token=0, max_length=8, temperature=0.8, stochastic=False, device=device)
    print("generated:", generated)
