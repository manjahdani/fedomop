# models/tabular_resnet_decomposable.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from fedomop.models.decomposable import ModelManager, ModelSplit
from fedomop.models.tabular import FederatedResNet, train, test


# -------------------- Body/Head split --------------------
class _ResnetBody(nn.Module):
    """Shared body for tabular Resnet.

    Assumes `Resnet` exposes:
      - `body`: nn.Module producing a hidden representation
    """

    def __init__(self, m: FederatedResNet):
        super().__init__()
        self.body = m.body

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class _ResnetHead(nn.Module):
    """Private head for tabular Resnet.

    Assumes `Resnet` exposes:
      - `head`: nn.Module mapping hidden -> output (e.g., logits)
    """

    def __init__(self, m: Resnet):
        super().__init__()
        self.head = m.head

    def forward(self, feat: Tensor) -> Tensor:
        return self.head(feat)


class ResnetSplit(ModelSplit):
    """FedPer split for tabular Resnet: BODY (shared) + HEAD (private)."""

    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        assert isinstance(model, FederatedResNet), "ResnetSplit expects a maidam.ml.models.tabular.Resnet"
        body = _ResnetBody(model)
        head = _ResnetHead(model)
        return body, head

    # ---- BODY param I/O (wire) ----
    def get_body_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.body.state_dict()

    def set_body_from_ndarrays(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        # If this looks like a full ModelSplit state_dict, extract the body part
        first_key = next(iter(state_dict))
        if first_key.startswith("_body."):
            prefix = "_body."
            body_state = OrderedDict(
                (k[len(prefix) :], v) for k, v in state_dict.items() if k.startswith(prefix)
            )
        else:
            body_state = state_dict  # assume body-only
        self.body.load_state_dict(body_state, strict=True)

    # ---- HEAD persistence (local/private) ----
    def body_state_dict(self) -> OrderedDict:
        return self.body.state_dict()

    def head_state_dict(self) -> OrderedDict:
        return self.head.state_dict()

    def load_head_state_dict(self, sd: OrderedDict) -> None:
        self.head.load_state_dict(sd, strict=True)


# -------------------- Manager --------------------
class ResnetManager(ModelManager):
    """Manager wrapping ResnetSplit for FedPer-like training using tabular Resnet's train/test."""

    def __init__(
        self,
        client_id: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        input_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.3,
        device: str = "cpu",
        # keep flexible: your tabular Resnet may accept additional kwargs
        resnet_kwargs: Dict[str, Union[int, float, str]] | None = None,
    ):
        self._trainloader = trainloader
        self._valloader = valloader
        self._device = device

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._n_blocks = n_blocks
        self._dropout = dropout
        self._resnet_kwargs = resnet_kwargs or {}

        super().__init__(client_id=client_id, config=None, model_split_class=ResnetSplit)

    def _create_model(self) -> nn.Module:
        # Must match maidam.ml.models.tabular.Resnet signature.
        return FederatedResNet(
            input_dim=self._input_dim,
            hidden_dim=self._hidden_dim,
            n_blocks=self._n_blocks,
            dropout=self._dropout,
            **self._resnet_kwargs,
        )

    def train(
        self,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        freeze_body: bool = False,
        freeze_head: bool = False,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        if freeze_body:
            self.model.disable_body()
        else:
            self.model.enable_body()

        if freeze_head:
            self.model.disable_head()
        else:
            self.model.enable_head()

        if self._input_dim == 12818:
            return train_h(
            net=self.model,  # ModelSplit is nn.Module
            trainloader=self._trainloader,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            device=self._device,
        )
        else:
            return train(
                net=self.model,  # ModelSplit is nn.Module
                trainloader=self._trainloader,
                epochs=epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                device=self._device,
            )

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        if self._input_dim == 12818:
            return test_h(self.model, self._valloader, self._device)
        else:
            return test(self.model, self._valloader, self._device)

    def train_dataset_size(self) -> int:
        return len(self._trainloader.dataset)

    def test_dataset_size(self) -> int:
        return len(self._valloader.dataset)

    def total_dataset_size(self) -> int:
        return self.train_dataset_size() + self.test_dataset_size()
