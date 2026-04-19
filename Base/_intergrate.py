from torch.nn import Module, ModuleList, Sequential
from torch import Tensor, stack, zeros, cat
from typing import Callable, Iterator
import math

class Parallel(Module):
    _modules: dict[str, Module] #type: ignore
    def __init__(self, *modules: Module):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
    def __iter__(self):
        return iter(self._modules.values())
    def __len__(self):
        return len(self._modules)
    def forward(self, *x: Tensor) -> Tensor:
        return cat([module(i) for module, i in zip(self, x)], dim=1)
class Loop(Module):
    def __init__(self, loop:int, module:Module) -> None:
        super().__init__()
        self.module = module
        self.loop = loop
    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.loop):
            x = self.module(x)
        return x
class Splitter(Module):
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
class Expand(Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "loop"):
        """Simmary

        Args:
            num (int): num_of_out_channels
            mode (str, optional): "loop", "zero". Defaults to "loop".
        """
        super().__init__()
        self.mode = mode
        self.n = math.ceil(out_channels/ in_channels)
        self.out_channels = out_channels
    def forward(self, x:Tensor) -> Tensor:
        x = x.repeat(1, self.n, 1, 1)
        if (self.mode=="loop"):
            return x[:, :self.out_channels, :, :]
        elif (self.mode == "zero"):
            x[:, :self.out_channels, :, :] = 0
            return x
        return zeros()
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
    _modules: dict[str, Module] #type: ignore
    def __init__(self, in_channels : int, indices: tuple[tuple[int,int], ...], *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.indices = indices
        self.in_channels = in_channels
    def __iter__(self):
        return iter(self._modules.values())
    def forward(self, x:Tensor):
        previous = 0
        out = []
        for (start,end), module in zip(self.indices, self):
            if (start>previous):
                out.append(x[:, previous:start, :, :])
            out.append(module(x[:, start:end, :, :]))
            previous = end
        if (previous<self.in_channels):
            out.append(x[:, previous:self.in_channels, :, :])
        return cat(out, dim=1)
            
class Mul(Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    def forward(self, x:Tensor) -> Tensor:
        return self.scale*x