from typing_extensions import Self

from torch.nn import Module, ModuleList, Sequential
from torch import Tensor, stack, zeros, cat, split
from typing import Callable, Iterator

class Splitter(Module):
    _modules: dict[str, Module]  # type: ignore[assignment]
    def __init__(self, *modules: Module | None):
        super().__init__()
        self._mmodule: list[Module | None] = []
        for idx, module in enumerate(modules):
            if (not module is None):
                self.add_module(str(idx), module)
            self._mmodule.append(module)
            
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x:Tensor) -> Tensor:
        return cat([module(x) if (not module is None) else x for module in self._mmodule], dim=1)
class Parallel(Module):
    def __init__(self, channels: int, module: Module):
        super().__init__()
        self.module = module
        self.channels = channels
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = x.view(-1, self.channels, H, W)
        x = self.module(x)
        return x.view(B, -1, H, W)
class Loop(Module):
    def __init__(self, module:Module, loop:int) -> None:
        super().__init__()
        self.module = module
        self.loop = loop
    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.loop):
            x = self.module(x)
        return x
class Merger(Module):
    _modules: dict[str, Module]  # type: ignore[assignment]
    def __init__(self, num_channels: tuple[int,...], *submodules: Module | None):
        super().__init__()
        self.num_channels = num_channels
        self._mmodules: list[Module | None] = []
        for idx, module in enumerate(submodules):
            if (not module is None):
                self.add_module(str(idx), module)
            self._mmodules.append(module)
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
    def __len__(self) -> int:
        return len(self._modules)
    def forward(self, x: Tensor) -> Tensor:
        previous = 0
        output = []
        for i, module in zip(self.num_channels, self._mmodules):
            sub = x[:, previous:previous+i, :, :]
            output.append(module(sub) if (not module is None) else sub)
            previous += i
        if (previous < x.shape[1]):
            output.append(x[:, previous:, :, :])
        return cat(output, dim=1)
class Expand(Module):
    def __init__(self, num: int):
        super().__init__()
        self.num = num
    def forward(self, x:Tensor) -> Tensor:
        return x.expand(x.shape[0], self.num*x.shape[1], x.shape[2], x.shape[3])
class View(Module):
    def __init__(self, channels: int, dim: int = 0):
        super().__init__()
        self.dim = dim
        self.channels = channels
    def forward(self, x:Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = x.view(B, )
        return x.view()
class BranchTrainning(Module):
    def __init__(self, tranning_module: Module | None, eval_module: Module | None):
        super().__init__()
        self.ls = [tranning_module, eval_module]
        self.module = tranning_module
    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.module = self.ls[not self.training]
        return self
    def forward(self, x:Tensor) -> Tensor:
        if (not self.module is None):
            return self.module(x)
        else:
            return x
class Mul(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    def forward(self, x:Tensor) ->Tensor:
        mods = split(x, self.channels)
        mods = stack(mods, dim=1).prod(dim=2)
        return mods
class SumChannels(Module):
    def __init__(self, out_channel: int):
        super().__init__()
        self.channels = out_channel
    def forward(self, x:Tensor) -> Tensor:
        mods = x.split(self.channels, dim=1)
        mods = stack(mods, dim=1).sum(dim=2)
        return mods