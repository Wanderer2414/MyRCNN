from torch.nn import Module, ModuleList, Sequential
from torch import Tensor, stack, sum
from typing import Callable, Iterator, Any

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
    def __init__(self, *modules: Module):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
    def forward(self, *x: Tensor) -> tuple[Tensor, ...]:
        return tuple([module(i) for module, i in zip(self.modules(), x)])
class Loop(Module):
    def __init__(self, init: Callable[[Any], None], condition: Callable[[Any], bool], inc: Callable[[Any], None], func: Module) -> None:
        super().__init__()
        self.init = init
        self.condition = condition
        self.inc = inc
        self.func = func
    def forward(self, x: Any) -> Any:
        self.init(x)
        while (self.condition(x)):
            x = self.func(x)
            self.inc(x)
        return x
class Merger(Module):
    def __init__(self, num_channels: tuple[int,...], *submodules: Module):
        super().__init__()
        self.num_channels = num_channels
        for idx, module in enumerate(submodules):
            self.add_module(str(idx), module)
    def forward(self, *x: Tensor) -> tuple[Tensor, ...]:
        previous = 0
        output: list[Tensor] = []
        for i, module in zip(self.num_channels, self.modules()):
            sub = x[previous:previous+i]
            output.append(module(sub))
            previous += i
        return tuple(output)
class Stack(Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
    def forward(self, arg: tuple[Tensor, ...]) -> Tensor:
        size = list[int](arg[0].shape)
        size[self.dim] *= len(arg)
        x = stack(arg, dim=self.dim)
        return x.view(*size)
class Select(Module):
    def __init__(self, idx: list[int]):
        super().__init__()
        self.idx = idx
    def forward(self, *x: Tensor) -> tuple[Tensor, ...]:
        output: list[Tensor] = []
        for i in self.idx:
            output.append(x[i])
        return tuple(output)
class Add(Module):
    def __init__(self):
        super().__init__()
    def forward(self, *x:Tensor) -> Tensor:
        return sum(*x)