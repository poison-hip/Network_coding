"""PyTorch RNN implementation for study.

Compared with the NumPy version in ``model.py``:

1. Parameters are managed by ``nn.Module``
2. Recurrent computation uses ``nn.RNN``
3. Autograd handles backward propagation
4. ``nn.CrossEntropyLoss`` computes the loss

This keeps the model readable while matching common PyTorch practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class Batch:
    """Simple container for sequence modeling data."""

    inputs: Tensor
    targets: Tensor


class RNNLanguageModel(nn.Module):
    """A compact RNN language model built from PyTorch nn modules.

    Data flow:
        token ids -> embedding -> RNN -> linear -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("vocab_size, embed_dim and hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use small random weights, similar to the handwritten version."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Tensor:
        """Initial hidden state with shape (num_layers, batch_size, hidden_dim)."""
        if device is None:
            device = self.embedding.weight.device
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        token_ids: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            token_ids: Shape (batch_size, seq_len)
            hidden: Shape (num_layers, batch_size, hidden_dim)

        Returns:
            logits: Shape (batch_size, seq_len, vocab_size)
            hidden_n: Final hidden state
        """
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape (batch_size, seq_len)")

        batch_size = token_ids.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, token_ids.device)

        x = self.embedding(token_ids)
        rnn_out, hidden_n = self.rnn(x, hidden)
        logits = self.output(rnn_out)
        return logits, hidden_n

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Average cross-entropy over all time steps."""
        if logits.dim() != 3 or targets.dim() != 2:
            raise ValueError("logits must be (batch_size, seq_len, vocab_size) and targets (batch_size, seq_len)")

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)
        return nn.functional.cross_entropy(logits_flat, targets_flat)

    @torch.no_grad()
    def predict(self, token_ids: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Return predicted token ids and final hidden state."""
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
        """Generate one token at a time."""
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
    """Build next-token prediction data from one token sequence."""
    if len(sequence) < 2:
        raise ValueError("sequence length must be at least 2")

    inputs = torch.tensor(sequence[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(sequence[1:], dtype=torch.long, device=device).unsqueeze(0)
    return Batch(inputs=inputs, targets=targets)


def train_one_epoch(
    model: RNNLanguageModel,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    clip_norm: Optional[float] = 5.0,
) -> dict:
    """Typical PyTorch training step."""
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

    model = RNNLanguageModel(
        vocab_size=4,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
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
