from torch.nn import Module
from torch import Tensor
from typing import Callable

class Parallel(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.sub_modules = modules
    def forward(self, x:Tensor) -> tuple[Tensor, ...]:
        return tuple([module(x) for module in self.sub_modules])
class Loop(Module):
    def __init__(self, init: Callable[[], ], condition: Callable[[], bool], inc: Callable[[], ], func: Callable[[Tensor], Tensor]) -> None:
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
class Sequential(Module):
    def __init__(self, *submodules:Module):
        super().__init__()
        self.submodules = submodules
    def forward(self, *args:Tensor) -> tuple[Tensor, ...]:
        x: Tensor | tuple[Tensor, ...] = args
        for module in self.submodules:
            x = module(*x)
            if (x is Tensor): 
                x = [args]
        return args
class Splitter(Module):
    def __init__(self, num_channels: tuple[int,...], *submodules: Module):
        super().__init__()
        self.num_channels = num_channels
        self.submodules = submodules
    def forward(self, *x: Tensor) -> tuple[Tensor, ...]:
        previous = 0
        output: list[Tensor] = []
        for i in range(len(self.num_channels)):
            sub = x[previous:previous+self.num_channels[i]]
            output.append(self.submodules[i](sub))
            previous += self.num_channels[i]
        return tuple(output)