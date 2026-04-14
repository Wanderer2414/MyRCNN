from torch.nn import Module, ModuleList, Sequential
from torch import Tensor, stack, zeros, cat
from typing import Callable, Iterator

class Splitter(Module):
    _modules: dict[str, Module]  # type: ignore[assignment]
    def __init__(self, *modules: Module):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
            
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x:Tensor) -> tuple[Tensor, ...]:
        return tuple(module(x) for module in self)
class Parallel(Module):
    def __init__(self, *modules: Module, loop: int = 1):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
    def forward(self, *x: Tensor) -> tuple[Tensor, ...]:
        return tuple([module(i) for module, i in zip(self.modules(), x)])
class Loop(Module):
    def __init__(self, init: Callable[[], None], condition: Callable[[], bool], inc: Callable[[], None], func: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.init = init
        self.condition = condition
        self.inc = inc
        self.func = func
    def forward(self, x: Tensor) -> Tensor:
        self.init()
        while (self.condition):
            x = self.func(x)
            self.inc()
        return x
class Merger(Module):
    _modules: dict[str, Module]  # type: ignore[assignment]
    def __init__(self, num_channels: tuple[int,...], *submodules: Module):
        super().__init__()
        self.num_channels = num_channels
        for idx, module in enumerate(submodules):
            self.add_module(str(idx), module)
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
    def __len__(self) -> int:
        return len(self._modules)
    def forward(self, x: Tensor) -> Tensor:
        previous = 0
        output = []
        for i, module in zip(self.num_channels, self):
            sub = x[:, previous:previous+i, :, :]
            output.append(module(sub))
            previous += i
        return cat(output, dim=1)
class Repeat(Module):
    def __init__(self, num: int):
        super().__init__()
        self.num = num
    def forward(self, x:Tensor) -> Tensor:
        return x.repeat(1, self.num, 1, 1)
class Stack(Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
    def forward(self, arg: tuple[Tensor, ...]) -> Tensor:
        size = list[int](arg[0].shape)
        size[self.dim] *= len(arg)
        x = stack(arg, dim=self.dim)
        return x.view(*size)