from typing import Callable, List, Optional
import torch


class MLP(torch.nn.Sequential):
    def __init__(
            self,
            num_neurons: List[int],
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        layers = []
        for c_in, c_out in zip(num_neurons[:-1], num_neurons[1:]):
            layers.append(torch.nn.Linear(c_in, c_out, bias=bias))
            layers.append(activation_layer())
            layers.append(torch.nn.Dropout(dropout))

        layers.pop()
        layers.pop()

        super().__init__(*layers)
