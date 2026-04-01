from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class Batch:
    inputs: Tensor
    targets: Tensor


class LSTMCell(nn.Module):
    """A single LSTM cell implemented from gate equations."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_i = nn.Parameter(torch.empty(input_dim + hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        self.W_f = nn.Parameter(torch.empty(input_dim + hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        self.W_o = nn.Parameter(torch.empty(input_dim + hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        self.W_g = nn.Parameter(torch.empty(input_dim + hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("W_"):
                # 权重参数用xavier初始化方式，每个w随机初始化学出不同特征并且训练稳定
                nn.init.xavier_uniform_(param)
            else:
                # 偏置参数用0初始化
                nn.init.zeros_(param)

        with torch.no_grad():
            # 给遗忘门一个偏向于多保留的倾向 sigmod(1) = 0.73
            self.b_f.fill_(1.0)

    def forward(self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """One recurrent step.

        Args:
            x_t: Shape (batch_size, input_dim)
            h_prev: Shape (batch_size, hidden_dim)
            c_prev: Shape (batch_size, hidden_dim)
        """
        combined = torch.cat([x_t, h_prev], dim=-1)

        # (batch_size, input_dim + hidden_dim) -> (batch_size, hidden_dim)
        i_t = torch.sigmoid(combined @ self.W_i + self.b_i)
        f_t = torch.sigmoid(combined @ self.W_f + self.b_f)
        o_t = torch.sigmoid(combined @ self.W_o + self.b_o)
        g_t = torch.tanh(combined @ self.W_g + self.b_g)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTMLanguageModel(nn.Module):
    """Language model built from a handwritten LSTM cell."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("vocab_size, embed_dim and hidden_dim must be positive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = LSTMCell(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        if device is None:
            device = self.embedding.weight.device
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h0, c0

    def forward(
        self,
        token_ids: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Manual unrolling over time.

        Args:
            token_ids: Shape (batch_size, seq_len)
            state: Tuple(h0, c0), each shape (batch_size, hidden_dim)
        """
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape (batch_size, seq_len)")

        batch_size, seq_len = token_ids.shape
        if state is None:
            h_t, c_t = self.init_state(batch_size, token_ids.device)
        else:
            h_t, c_t = state

        embeddings = self.embedding(token_ids)
        outputs = []

        for t in range(seq_len):
            x_t = embeddings[:, t, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            logits_t = self.output(h_t)
            outputs.append(logits_t)

        logits = torch.stack(outputs, dim=1)
        return logits, (h_t, c_t)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.dim() != 3 or targets.dim() != 2:
            raise ValueError("logits must be (batch_size, seq_len, vocab_size) and targets (batch_size, seq_len)")

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)
        return nn.functional.cross_entropy(logits_flat, targets_flat)

    @torch.no_grad()
    def predict(
        self,
        token_ids: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        self.eval()
        logits, final_state = self.forward(token_ids, state)
        predictions = torch.argmax(logits, dim=-1)
        return predictions, final_state

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
        state = self.init_state(batch_size=1, device=device)
        generated = [start_token]

        for _ in range(max_length - 1):
            logits, state = self.forward(current, state)
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
    model: LSTMLanguageModel,
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

    model = LSTMLanguageModel(
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
