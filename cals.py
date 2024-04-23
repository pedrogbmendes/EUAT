import logging
import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from typing import Tuple
from torch import Tensor
from torch import distributed as dist

logger = logging.getLogger(__name__)




def p2(h: Tensor, lambd: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor]:
    y_sup = lambd * h + lambd * rho * (h ** 2) + 1 / 6 * (rho ** 2) * (h ** 3)
    y_inf = lambd * h / (1 - rho * h.clamp(max=0))

    grad_y_sup = lambd + 2 * lambd * rho * h + 1 / 2 * (rho ** 2) * (h ** 2)
    grad_y_inf = lambd / (1 - rho * h.clamp(max=0)) ** 2

    sup = h >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def p3(h: Tensor, lambd: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor]:
    if lambd.ndim == 1:
        lambd = lambd.unsqueeze(dim=0).expand(h.shape)

    if isinstance(rho, torch.Tensor) and rho.ndim == 1:
        rho = rho.unsqueeze(dim=0).expand(h.shape)

    y_sup = lambd * h + lambd * rho * (h ** 2)
    y_inf = lambd * h / (1 - rho * h.clamp(max=0))

    grad_y_sup = lambd + 2 * lambd * rho * h
    grad_y_inf = lambd / (1 - rho * h.clamp(max=0)) ** 2

    sup = h >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def phr(h: Tensor, lambd: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor]:
    x = lambd + rho * h
    y_sup = 1 / (2 * rho) * (x ** 2 - lambd ** 2)
    y_inf = - 1 / (2 * rho) * (lambd ** 2)

    grad_y_sup = x
    grad_y_inf = torch.zeros_like(h)

    sup = x >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def relu(h: Tensor, lambd: Tensor, *arg) -> Tuple[Tensor, Tensor]:
    y_sup = lambd * h
    y_inf = torch.zeros_like(h)

    grad_y_sup = lambd
    grad_y_inf = torch.zeros_like(h)

    sup = h >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def get_penalty_func(name):
    all_penalties = {
        "p2": p2,
        "p3": p3,
        "phr": phr,
    }

    return all_penalties[name]



def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


class AugLagrangian:
    """Augmented Lagrangian Method

    Args:
        penalty_func (Callable):
            Penalty-Lagrangian to use.
        lambda_buffer_size (int): buffer size of penalty multipliers
            It depends on the used sampler in the dataloader.           
            To cover all the cases, we can set it as train_size + val_size.
            It is not ideal bc we don't use the buffer corresponding to the val indices in fact.
            Note we should return the index of the training samples in dataloader.
        lambda_init (float, optional): Initial value of the penalty multipliers. Defaults to 0..
        rho_init (float, optional): Initial value of the penalty parameter. Defaults to 1.0.
        gamma (float, optional): Increase rate of penalty paramter rho. Defaults to 1.2.
        tao (float, optional): constraint improvement rate. Defaults to 2.0.
        start_epoch (int, optional): The start epoch to enforce the constraint. Defaults to 1.
    """
    def __init__(
        self,
        num_classes: int = 10,
        margin: float = 10.,
        penalty: str = "p2",
        lambd_min: float = 10e-6,
        lambd_max: float = 10e6,
        lambd_step: int = 1,
        classwise: bool = False,
        rho_init: float = 1.0,
        rho_max: float = 10.0,
        rho_update: str = "",
        rho_step: int = 1,
        gamma: float = 1.2,
        tao: float = 0.9,
        start_epoch: int = 1,
        device: torch.device = "cuda:0",
        normalize: bool = True
    ):
        self.num_classes = num_classes
        self.margin = margin
        self.penalty_func = get_penalty_func(penalty)
        self.lambd_min = lambd_min
        self.lambd_max = lambd_max
        self.lambd_step = lambd_step
        self.classwise = classwise
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.rho_update = rho_update
        self.rho_step = rho_step
        self.rho = rho_init
        self.gamma = gamma
        self.tao = tao
        self.start_epoch = start_epoch
        self.device = device
        self.normalize = normalize


        #self.lambd_buffer_size = num_classes
        self.lambd_buffer = (
            lambd_min * torch.ones(self.lambd_buffer_size, self.num_classes)
        )

        if self.rho_update == "with_train":
            self.rho_buffer = (
                rho_init * torch.ones(self.lambd_buffer_size, self.num_classes)
            )
            self.prev_penalty = torch.zeros(self.lambd_buffer_size, self.num_classes)
        elif self.rho_update == "with_val":
            self.prev_penalty = None

    def get_constraints(self, logits, dim: int = -1):
        max_values = logits.amax(dim=dim, keepdim=True)
        diff = max_values - logits
        constraints = diff - self.margin
        if self.normalize:
            constraints = constraints / self.margin
        return constraints

    def get(self, logits: torch.Tensor, indices: torch.Tensor, epoch: int) -> List[torch.Tensor]:
        h = self.get_constraints(logits)
        lambd = self.lambd_buffer[indices].to(self.device)

        if self.rho_update == "with_train":
            rho = self.rho_buffer[indices].to(self.device)
        else:
            rho = self.rho

        p, _ = self.penalty_func(h, lambd, rho)
        penalty = p.mean()
        constraint = h.mean()
        if self.rho_update == "with_train" and epoch is not None and (epoch + 1) % self.rho == 0:
            prev_p = self.prev_penalty[indices]
            p = p.detach().cpu()
            rho = rho.detach().cpu()
            rho = torch.where(
                (prev_p > 0) & (p > self.tao * prev_p),
                (self.gamma * rho).clamp(max=self.rho_max),
                rho
            )
            self.prev_penalty[indices] = p
            self.rho_buffer[indices] = rho
        return penalty, constraint

    def update_lambd(self, logits: torch.Tensor, indices: torch.Tensor, epoch):
        if (epoch + 1) % self.lambd_step == 0:
            h = self.get_constraints(logits)
            lambd = self.lambd_buffer[indices].to(self.device)

            _, grad_p = self.penalty_func(h, lambd, self.rho)
            lambd = torch.clamp(grad_p, min=self.lambd_min, max=self.lambd_max)
            self.lambd_buffer[indices] = lambd.detach().cpu()

    def update_rho_by_val(self, val_penalty, epoch):
        if self.rho_update == "with_val" and (epoch + 1) % self.rho_step == 0:
            if self.prev_penalty is not None:
                if (
                    self.prev_penalty > 0
                    and val_penalty > 0
                    and val_penalty > self.tao * self.prev_penalty
                ):
                    self.rho = min(self.rho_max, self.rho * self.gamma)
                    logger.info("Adjusting rho in AugLagrangian to {}".format(self.rho))
            self.prev_penalty = val_penalty

    def set_train_indices(self, data_loader):
        self.train_indices = None
        for i, (_, _, indices) in enumerate(data_loader):
            if self.train_indices is None:
                self.train_indices = indices
            else:
                self.train_indices = torch.cat((self.train_indices, indices))

    def plot_lambd_hist(self):
        lambd = self.lambd_buffer[self.train_indices]

        fig = plt.figure("lambda hist")
        ax = fig.add_subplot()
        ax.hist(lambd.numpy(), bins=np.arange(11), density=True)

        ax.set_xlabel("lambda values")

        plt.tight_layout()

        return fig

    def get_rho(self):
        if self.rho_update == "with_val":
            return self.rho
        else:
            return self.rho.mean().item()

    def get_lambd_metric(self):
        lambd = self.lambd_buffer[self.train_indices]

        return lambd.mean().item(), lambd.max().item()
    



class AugLagrangianClass(AugLagrangian):
    def __init__(
        self,
        num_classes: int = 10,
        margin: float = 10,
        penalty: str = "p2",
        lambd_min: float = 1e-6,
        lambd_max: float = 1e6,
        lambd_step: int = 1,
        rho_min: float = 1,
        rho_max: float = 10,
        # rho_update: bool = False,
        rho_step: int = -1,
        gamma: float = 1.2,
        tao: float = 0.9,
        normalize: bool = True
    ):
        assert penalty in ("p2", "p3", "phr", "relu"), f"invalid penalty: {penalty}"
        self.num_classes = num_classes
        self.margin = margin
        self.penalty_func = get_penalty_func(penalty)
        self.lambd_min = lambd_min
        self.lambd_max = lambd_max
        self.lambd_step = lambd_step
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_update = rho_step > 0
        self.rho_step = rho_step
        self.gamma = gamma
        self.tao = tao
        self.normalize = normalize

        self.lambd = self.lambd_min * torch.ones(self.num_classes, requires_grad=False).cuda()
        # self.rho = self.rho_min
        # class-wise rho
        self.rho = self.rho_min * torch.ones(self.num_classes, requires_grad=False).cuda()
        # for updating rho
        self.prev_constraints, self.curr_constraints = None, None

    def get(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits.movedim(1, -1)  # move class dimension to last

        h = self.get_constraints(logits)
        p, _ = self.penalty_func(h, self.lambd, self.rho)
        # penalty = p.sum(dim=-1).mean()  # sum over classes and average over samples (and possibly pixels)
        penalty = p.mean()
        constraint = h.mean()
        return penalty, constraint

    def reset_update_lambd(self, epoch):
        # if (epoch + 1) % self.lambd_step == 0:
        self.grad_p_sum = torch.zeros_like(self.lambd)
        self.sample_num = 0
        self.curr_constraints = torch.zeros_like(self.rho)

    def update_lambd(self, logits, epoch):
        """update lamdb based on the gradeint on the logits
        """
        # if (epoch + 1) % self.lambd_step == 0:
        logits = logits.movedim(1, -1)  # move class dimension to last

        h = self.get_constraints(logits)
        _, grad_p = self.penalty_func(h, self.lambd, self.rho)
        grad_p = torch.clamp(grad_p, min=self.lambd_min, max=self.lambd_max)
        grad_p = grad_p.flatten(start_dim=0, end_dim=-2)
        self.grad_p_sum += grad_p.sum(dim=0)
        self.sample_num += grad_p.shape[0]
        h = h.flatten(start_dim=0, end_dim=-2)
        self.curr_constraints += h.sum(dim=0)

    def set_lambd(self, epoch):
        if (epoch + 1) % self.lambd_step == 0:
            grad_p_mean = self.grad_p_sum / self.sample_num
            if dist.is_initialized():
                grad_p_mean = reduce_tensor(grad_p_mean, dist.get_world_size())
            self.lambd = torch.clamp(grad_p_mean, min=self.lambd_min, max=self.lambd_max).detach()

    def update_rho(self, epoch):
        if self.rho_update:
            self.curr_constraints = self.curr_constraints / self.sample_num
            if dist.is_initialized():
                self.curr_constraints = reduce_tensor(self.curr_constraints, dist.get_world_size())

            if (epoch + 1) % self.rho_step == 0 and self.prev_constraints is not None:
                # increase rho if the constraint became unsatisfied or didn't decrease as expected
                self.rho = torch.where(
                    self.curr_constraints > (self.prev_constraints.clamp(min=0) * self.tao),
                    self.gamma * self.rho,
                    self.rho
                )
                self.rho = torch.clamp(self.rho, min=self.rho_min, max=self.rho_max).detach()

            self.prev_constraints = self.curr_constraints

    def get_lambd_metric(self):
        lambd = self.lambd

        return lambd.mean().item(), lambd.max().item()

    def get_rho_metric(self):
        return self.rho.mean().item(), self.rho.max().item()