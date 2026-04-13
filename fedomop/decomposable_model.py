"""Abstract class for splitting a model into body and head."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple, Type, Union
from omegaconf import DictConfig
import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from torch.utils.data import DataLoader


from fedomop.model import ResMLP, train, test


class ModelSplit(ABC, nn.Module):
    """Abstract class for splitting a model into body and head."""

    def __init__(
        self,
        model: nn.Module,
    ):
        """Initialize the attributes of the model split.

        Args:
            model: dict containing the vocab sizes of the input attributes.
        """
        super().__init__()

        self._body, self._head = self._get_model_parts(model)

    @abstractmethod
    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return the body and head of the model.

        Args:
            model: model to be split into head and body

        Returns
        -------
            Tuple where the first element is the body of the model
            and the second is the head.
        """

    @property
    def body(self) -> nn.Module:
        """Return model body."""
        return self._body

    @body.setter
    def body(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model body.

        Args:
            state_dict: dictionary of the state to set the model body to.
        """
        self.body.load_state_dict(state_dict, strict=True)

    @property
    def head(self) -> nn.Module:
        """Return model head."""
        return self._head

    @head.setter
    def head(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model head.

        Args:
            state_dict: dictionary of the state to set the model head to.
        """
        self.head.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters (without fixed head).

        Returns
        -------
            Body and head parameters
        """
        return [
            val.cpu().numpy()
            for val in [
                *self.body.state_dict().values(),
                *self.head.state_dict().values(),
            ]
        ]

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        ordered_state_dict = OrderedDict(self.state_dict().copy())
        # Update with the values of the state_dict
        ordered_state_dict.update(dict(state_dict.items()))
        self.load_state_dict(ordered_state_dict, strict=False)

    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = True

    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = True

    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = False

    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = False

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the body and the head."""
        x = self.body(inputs)
        return self.head(x)


class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        model_split_class: Type[Any],  # ModelSplit
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            model_split_class: Class to be used to split the model into body and head\
                (concrete implementation of ModelSplit).
        """
        super().__init__()

        self.client_id = client_id
        self.config = config
        self._model = model_split_class(self._create_model())

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be splitted into head and body."""

    @abstractmethod
    def train(
        self,
        epochs: int = 1,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Args:
            epochs: number of training epochs.

        Returns
        -------
            Dict containing the train metrics.
        """

    @abstractmethod
    def test(
        self,
    ) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Returns
        -------
            Dict containing the test metrics.
        """

    @abstractmethod
    def train_dataset_size(self) -> int:
        """Return train data set size."""

    @abstractmethod
    def test_dataset_size(self) -> int:
        """Return test data set size."""

    @abstractmethod
    def total_dataset_size(self) -> int:
        """Return total data set size."""

    @property
    def model(self) -> nn.Module:
        """Return model."""
        return self._model




# -------------------- Body/Head split --------------------
class _ResnetBody(nn.Module):
    """Shared body for tabular Resnet.

    Assumes `Resnet` exposes:
      - `body`: nn.Module producing a hidden representation
    """

    def __init__(self, m: ResMLP):
        super().__init__()
        self.body = m.body

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class _ResnetHead(nn.Module):
    """Private head for tabular Resnet.

    Assumes `Resnet` exposes:
      - `head`: nn.Module mapping hidden -> output (e.g., logits)
    """

    def __init__(self, m: ResMLP):
        super().__init__()
        self.head = m.head

    def forward(self, feat: Tensor) -> Tensor:
        return self.head(feat)


class ResnetSplit(ModelSplit):
    """FedPer split for tabular Resnet: BODY (shared) + HEAD (private)."""

    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        assert isinstance(model, ResMLP), f"Expected ResMLP, got {type(model)}"
        body = _ResnetBody(model)
        head = _ResnetHead(model)
        return body, head

    # ---- BODY param I/O (wire) ----
    def get_body_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.body.state_dict()


    def set_body_from_ndarrays(self, state_dict):
        body_state = OrderedDict(
            (k, v) for k, v in state_dict.items() if k.startswith("body.")
        )
        self.body.load_state_dict(body_state, strict=True)
    # def set_body_from_ndarrays(self, state_dict: Mapping[str, torch.Tensor]) -> None:
    #     # If this looks like a full ModelSplit state_dict, extract the body part
    #     first_key = next(iter(state_dict))
    #     if first_key.startswith("_body."):
    #         prefix = "_body."
    #         body_state = OrderedDict(
    #             (k[len(prefix) :], v) for k, v in state_dict.items() if k.startswith(prefix)
    #         )
    #     else:
    #         body_state = state_dict  # assume body-only
    #     self.body.load_state_dict(body_state, strict=True)

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
        return ResMLP(
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
        return train(
            net = self.model,  # ModelSplit is nn.Module
            trainloader = self._trainloader,
            epochs = epochs,
            lr = lr,
            weight_decay = weight_decay,
            device = self._device,
        )

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        return test(self.model, self._valloader, self._device)

    def train_dataset_size(self) -> int:
        return len(self._trainloader.dataset)

    def test_dataset_size(self) -> int:
        return len(self._valloader.dataset)

    def total_dataset_size(self) -> int:
        return self.train_dataset_size() + self.test_dataset_size()

