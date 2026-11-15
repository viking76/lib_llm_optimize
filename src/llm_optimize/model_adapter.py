"""Model adapters: extract answers and activations from a model under test.

Two adapters are provided:

- :class:`MockModel` — a tiny deterministic numpy model used in tests and as a
  runnable example without any heavy dependency.
- :class:`TransformersAdapter` — hooks a Hugging Face causal LM and captures
  per-layer MLP activations, in the spirit of Neuroscope-style neuron
  inspection.

Any custom model only has to implement :class:`ModelAdapter`.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np

from .types import Question

ActivationMap = Dict[int, np.ndarray]


class ModelAdapter(Protocol):
    """Minimal contract the pipeline needs from a model."""

    n_layers: int

    def answer(self, question: Question) -> str:
        """Return the model answer for a question."""
        ...

    def activations(self, texts: Sequence[str]) -> ActivationMap:
        """Return mean activation per layer: {layer_index: array[n_neurons]}."""
        ...

    def per_question_activations(self, texts: Sequence[str]) -> Dict[int, np.ndarray]:
        """Return per-question activations: {layer: array[n_questions, n_neurons]}."""
        ...

    def copy(self) -> "ModelAdapter":
        """Return a deep, independent copy so patches can be tested safely."""
        ...

    def apply_scale(self, layer: int, neuron: int, scale: float) -> None:
        """Scale the outgoing weights of one neuron in place."""
        ...


def _bow(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic hashed bag-of-words feature vector in [0, 1]."""
    vec = np.zeros(dim, dtype=np.float64)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class MockModel:
    """A toy 2-layer MLP model with an injectable "quirk" neuron.

    The quirk neuron over-activates on texts containing trigger words and
    flips the answer to a wrong token, which mimics the failure mode the
    library is designed to diagnose and repair.
    """

    def __init__(
        self,
        n_layers: int = 2,
        n_neurons: int = 32,
        feature_dim: int = 64,
        trigger_words: Sequence[str] = (),
        quirk_layer: int = 0,
        seed: int = 0,
    ) -> None:
        if quirk_layer != 0:
            raise ValueError("MockModel injects the quirk neuron on layer 0")
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.feature_dim = feature_dim
        self.trigger_words = list(trigger_words)
        self.quirk_layer = quirk_layer
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = [
            rng.normal(0.0, 0.5, size=(n_neurons, feature_dim))
        ] + [
            rng.normal(0.0, 0.5, size=(n_neurons, n_neurons))
            for _ in range(n_layers - 1)
        ]
        self.quirk_neuron = 0
        self.quirk_row = np.zeros(feature_dim)
        if self.trigger_words:
            for word in self.trigger_words:
                self.quirk_row += _bow(word, feature_dim)
            norm = np.linalg.norm(self.quirk_row)
            if norm > 0:
                self.quirk_row /= norm
            self.weights[quirk_layer][self.quirk_neuron] = self.quirk_row * 7.0

    def _forward(self, text: str) -> Tuple[ActivationMap, float]:
        features = _bow(text, self.feature_dim)
        acts: ActivationMap = {}
        hidden = features
        for layer, W in enumerate(self.weights):
            pre = W @ hidden
            act = np.maximum(pre, 0.0)
            acts[layer] = act
            hidden = act / (np.linalg.norm(act) + 1e-9)
        quirk_act = float(acts[self.quirk_layer][self.quirk_neuron])
        return acts, quirk_act

    def answer(self, question: Question) -> str:
        _, quirk_act = self._forward(question.text)
        if quirk_act > 2.0:
            return "WRONG"
        return question.expected_answer

    def per_question_activations(self, texts: Sequence[str]) -> Dict[int, np.ndarray]:
        result: Dict[int, List[np.ndarray]] = {
            layer: [] for layer in range(self.n_layers)
        }
        for text in texts:
            acts, _ = self._forward(text)
            for layer, act in acts.items():
                result[layer].append(act)
        return {
            layer: np.stack(rows) if rows else np.zeros((0, self.n_neurons))
            for layer, rows in result.items()
        }

    def activations(self, texts: Sequence[str]) -> ActivationMap:
        per_question = self.per_question_activations(texts)
        return {
            layer: values.mean(axis=0) if values.size else np.zeros(self.n_neurons)
            for layer, values in per_question.items()
        }

    def copy(self) -> "MockModel":
        clone = MockModel(
            n_layers=self.n_layers,
            n_neurons=self.n_neurons,
            feature_dim=self.feature_dim,
            trigger_words=self.trigger_words,
            quirk_layer=self.quirk_layer,
        )
        clone.weights = [W.copy() for W in self.weights]
        return clone

    def apply_scale(self, layer: int, neuron: int, scale: float) -> None:
        self.weights[layer][neuron, :] *= scale


class TransformersAdapter:
    """Hook a Hugging Face causal LM and expose its MLP activations.

    Requires the optional dependencies: ``pip install llm-optimize[hf]``
    (``torch`` and ``transformers``).

    ``apply_scale`` edits the ``mlp.down_proj`` weight row of the given
    neuron, i.e. exactly the outgoing weights of one intermediate neuron.
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()
        self.device = device
        self._mlp_modules = self._locate_mlp_modules()
        self.n_layers = len(self._mlp_modules)
        self._hooks: list = []
        self._captured: Dict[int, "torch.Tensor"] = {}

    def _locate_mlp_modules(self) -> list:
        modules = []
        for module in self.model.modules():
            if type(module).__name__ == "MLP" and hasattr(module, "down_proj"):
                modules.append(module)
        if not modules:
            raise ValueError(
                "No decoder MLP layers found; this adapter targets standard "
                "causal LMs (GPT-2, Pythia, OLMo, Llama-like architectures)."
            )
        return modules

    def _install_hooks(self) -> None:
        def make_hook(layer: int):
            def hook(_module, _inputs, output):
                self._captured[layer] = output.detach()

            return hook

        for layer, mlp in enumerate(self._mlp_modules):
            self._hooks.append(mlp.register_forward_hook(make_hook(layer)))

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _encode(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

    def answer(self, question: Question) -> str:
        import torch

        inputs = self._encode([question.text])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=32, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def per_question_activations(self, texts: Sequence[str]) -> Dict[int, np.ndarray]:
        self._captured.clear()
        self._install_hooks()
        try:
            inputs = self._encode(texts)
            with self._torch.no_grad():
                self.model(**inputs)
        finally:
            self._remove_hooks()
        result: Dict[int, np.ndarray] = {}
        for layer, tensor in self._captured.items():
            act = tensor.abs().mean(dim=1).cpu().numpy()
            result[layer] = act
        return result

    def activations(self, texts: Sequence[str]) -> ActivationMap:
        per_question = self.per_question_activations(texts)
        return {
            layer: values.mean(axis=0) if values.size else np.zeros(values.shape[1])
            for layer, values in per_question.items()
        }

    def copy(self) -> "TransformersAdapter":
        clone = TransformersAdapter.__new__(TransformersAdapter)
        clone._torch = self._torch
        clone.tokenizer = self.tokenizer
        clone.model = type(self.model).from_config(
            self.model.config
        ).to(device=self.device, dtype=next(self.model.parameters()).dtype)
        clone.model.load_state_dict(
            {k: v.clone() for k, v in self.model.state_dict().items()}
        )
        clone.model.eval()
        clone.device = self.device
        clone._mlp_modules = clone._locate_mlp_modules()
        clone._hooks = []
        clone._captured = {}
        return clone

    def apply_scale(self, layer: int, neuron: int, scale: float) -> None:
        mlp = self._mlp_modules[layer]
        with self._torch.no_grad():
            mlp.down_proj.weight[neuron, :] *= scale
