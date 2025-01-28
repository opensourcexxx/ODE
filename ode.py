from typing import Union, Tuple, Optional, Any, Callable
import math
from typing_extensions import Literal
import numpy as np
import eagerpy as ep
import logging
import torch.nn.functional as F

from foolbox.devutils import flatten
from foolbox.devutils import atleast_kd

from foolbox.types import Bounds

from foolbox.models import Model

from foolbox.criteria import Criterion

from foolbox.distances import l2, linf

from foolbox.blended_noise import LinearSearchBlendedUniformNoiseAttack

from foolbox import MinimizationAttack
from foolbox import T
from foolbox import get_criterion
from foolbox import get_is_adversarial
from foolbox import raise_if_kwargs


class mode(MinimizationAttack):
    distance = l2

    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 25000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        source_step_convergance: float = 1e-7,
        update_stats_every_k: int = 5,
        k_refer:float = 5.,
        cv:float = 0.1,
        cc:float = 0.5,
        Orthogonal_setting:bool = True,
        Enlarge_setting:bool = True,
        Enlarge_k:float=0.2,
        query_nums_limit:int =100000,
        constraint:str = "l2",
        args = None
    ):
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.init_attack = init_attack
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergance = source_step_convergance
        self.update_stats_every_k = update_stats_every_k
        self.k_refer = k_refer
        self.cv = cv
        self.cc = cc
        self.Orthogonal_setting = Orthogonal_setting
        self.Enlarge_setting = Enlarge_setting
        self.Enlarge_k = Enlarge_k
        self.query_nums_limit = query_nums_limit
        self.args = args
        self.constraint = constraint
        self.gamma = 1.0
        
        assert constraint in ("l2", "linf")
        if constraint == "l2":
            self.distance = l2
        else:
            self.distance = linf
        
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50,directions=10000)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            adv = init_attack.run(
                model, originals, criterion, early_stop=early_stop
            )
        else:
            adv = ep.astensor(starting_points)

        is_adv = is_adversarial(adv)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(
                    f"init_attack failed for {failed} of {len(is_adv)} inputs"
                )
            else:
                raise ValueError(
                    f"{failed} of {len(is_adv)} starting_points are not adversarial"
                )
        del starting_points
        
        # Project the initialization to the boundary.
        adv = self._binary_search(is_adversarial, originals, adv)

        N = len(originals)
        if self.args.Reduction_setting:
            _, _, h, w = originals.shape
            reduced_originals = F.interpolate(originals.raw, size=(h//self.args.reduce_r, w//self.args.reduce_r), mode='bilinear', align_corners=False)
            reduced_originals = ep.astensors(reduced_originals)[0]
            reduced_adv = F.interpolate(adv.raw, size=(h//self.args.reduce_r, w//self.args.reduce_r), mode='bilinear', align_corners=False)
            reduced_adv = ep.astensors(reduced_adv)[0]
        base_tensor_type = originals
        ndim = originals.ndim
        spherical_steps = ep.ones(base_tensor_type, N) * self.spherical_step
        source_steps = ep.ones(base_tensor_type, N) * self.source_step
        
        # source_mu = ep.zeros(base_tensor_type, N) * 0.5
        if self.args.Reduction_setting:
            C = ep.ones_like(reduced_originals)
            P_C = ep.zeros_like(reduced_originals)
        else:
            C = ep.ones_like(originals)
            P_C = ep.zeros_like(originals)
        stats_spherical_adversarial = ArrayQueue(maxlen=int(self.k_refer), N=N)
        stats_step_adversarial = ArrayQueue(maxlen=int(self.k_refer), N=N)
        bounds = model.bounds
        self.attack_process = []
        for step in range(1, self.steps + 1):
            # print(f"step:{step}")
            converged = source_steps < self.source_step_convergance
            # if converged.all():
            #     break  # pragma: no cover
            converged = atleast_kd(converged, ndim)

            # only check spherical candidates every k steps
            check_spherical_and_update_stats = step % self.update_stats_every_k == 0
            if self.args.Reduction_setting:
                candidates, spherical_candidates, C_candidates, P_C_candidates = draw_proposals(
                    bounds,
                    reduced_originals,
                    reduced_adv,
                    spherical_steps,
                    source_steps,
                    C,
                    P_C,
                    self.cc,
                    self.cv,
                    self.Orthogonal_setting
                )
                candidates = F.interpolate(candidates.raw, size=(h, w), mode='bilinear', align_corners=False)
                candidates = ep.astensors(candidates)[0]    
            else:
                candidates, spherical_candidates, C_candidates, P_C_candidates = draw_proposals(
                    bounds,
                    originals,
                    adv,
                    spherical_steps,
                    source_steps,
                    C,
                    P_C,
                    self.cc,
                    self.cv,
                    self.Orthogonal_setting
                )
            is_adv = is_adversarial(candidates)

            spherical_is_adv: Optional[ep.Tensor]
            if check_spherical_and_update_stats:
                # spherical_is_adv = is_adversarial(spherical_candidates)
                # stats_spherical_adversarial.append(spherical_is_adv)
                if self.Orthogonal_setting:
                    if self.args.Reduction_setting:
                        spherical_candidates = F.interpolate(spherical_candidates.raw, size=(h, w), mode='bilinear', align_corners=False)
                        spherical_candidates = ep.astensors(spherical_candidates)[0]
                    spherical_is_adv = is_adversarial(spherical_candidates)
                    stats_spherical_adversarial.append(spherical_is_adv)
                else:
                    # spherical_is_adv = None
                    spherical_is_adv = is_adv
                    stats_spherical_adversarial.append(spherical_is_adv)
                stats_step_adversarial.append(is_adv)
            else:
                spherical_is_adv = None

            distances = ep.norms.l2(flatten(originals - candidates+1e-5), axis=-1)
            closer = distances < ep.norms.l2(flatten(originals - adv+1e-5), axis=-1)
            # if self.constraint == "l2":
            #     distances = ep.norms.l2(flatten(originals - candidates+1e-5), axis=-1)
            #     closer = distances < ep.norms.l2(flatten(originals - adv+1e-5), axis=-1)
            # elif self.constraint == 'linf':
            #     distances = ep.norms.linf(flatten(originals - candidates+1e-5), axis=-1)
            #     closer = distances < ep.norms.linf(flatten(originals - adv+1e-5), axis=-1)
            is_better_adv = ep.logical_and(is_adv, closer)
            is_better_adv = atleast_kd(is_better_adv, ndim)

            cond = converged.logical_not().logical_and(is_better_adv)
            adv = ep.where(cond, candidates, adv)
            C = ep.where(cond, C_candidates, C)
            P_C = ep.where(cond, P_C_candidates, P_C)
            
            if check_spherical_and_update_stats:
                
                # if self.Orthogonal_setting:
                # 这部分Dong也是没有的
                if self.args.self_adaptive_spherical_step:
                    full = stats_spherical_adversarial.isfull()
                    if full.any():
                        probs = stats_spherical_adversarial.mean()
                        step_adaptation = ep.exp((probs-0.5)/0.5/3)
                        
                        cond1 = ep.logical_and(probs > 0.5, full)
                        spherical_steps = ep.where(cond1, spherical_steps * step_adaptation, spherical_steps)
                        # source_steps = ep.where(cond1, source_steps * step_adaptation, source_steps)
                        
                        cond2 = ep.logical_and(probs < 0.2, full)
                        is_hit = spherical_steps.uniform(cond2.shape) < self.Enlarge_k
                        if self.Enlarge_setting:
                            spherical_steps = ep.where(cond2,  ep.where(is_hit, spherical_steps / step_adaptation, spherical_steps * step_adaptation), spherical_steps)
                        else:
                            spherical_steps = ep.where(cond2,  spherical_steps * step_adaptation, spherical_steps)
                        
                        stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))
                        # c_success_queue = ep.where(ep.logical_or(cond1, cond2), ep.zeros_like(c_success_queue), c_success_queue)
                        C,P_C = clear_c_pc(atleast_kd(ep.logical_or(cond1, cond2),ndim),C,P_C)

                full = stats_step_adversarial.isfull()
                if full.any():
                    probs = stats_step_adversarial.mean()
                    step_adaptation = ep.exp((probs-0.2)/0.8/3)
                    cond1 = ep.logical_and(probs > 0.25, full)
                    source_steps = ep.where(cond1, source_steps * step_adaptation, source_steps)
                    cond2 = ep.logical_and(probs < 0.1, full)
                    source_steps = ep.where(cond2, source_steps * step_adaptation, source_steps)
                    
                    stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
                    # c_success_queue = ep.where(ep.logical_or(cond1, cond2), ep.zeros_like(c_success_queue),c_success_queue)
                    C, P_C = clear_c_pc(atleast_kd(ep.logical_or(cond1, cond2), ndim), C, P_C)
            if model.query_nums > self.query_nums_limit:
                break
            
            # Project the initialization to the boundary.
            # adv = self._binary_search(is_adversarial, originals, adv)
            
            distances = self.distance(originals, adv)
            # self.attack_process.append({"distances":distances.raw.item(),"query_nums":model.query_nums})
            self.attack_process.append({"distances":distances.raw.detach().cpu().numpy(),"query_nums":model.query_nums})
            
        return restore_type(adv)
    
    def _binary_search(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        originals: ep.Tensor,
        perturbed: ep.Tensor,
    ) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        d = int(np.prod(perturbed.shape[1:]))
        if self.constraint == "linf":
            highs = linf(originals, perturbed)

            # TODO: Check if the threshold is correct
            #  empirically this seems to be too low
            thresholds = highs * self.gamma / (d * d)
        else:
            highs = ep.ones(perturbed, len(perturbed))
            thresholds = highs * self.gamma / (d * math.sqrt(d))

        lows = ep.zeros_like(highs)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs

        while ep.any(highs - lows > thresholds):
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = is_adversarial(mids_perturbed)

            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids

            if reached_numerical_precision:
                # TODO: warn user
                break

        res = self._project(originals, perturbed, highs)

        return res
    
    def _project(
        self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor
    ) -> ep.Tensor:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            originals: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilons: A batch of norm values to project to.
        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        epsilons = atleast_kd(epsilons, originals.ndim)
        if self.constraint == "linf":
            perturbation = perturbed - originals

            # ep.clip does not support tensors as min/max
            clipped_perturbed = ep.where(
                perturbation > epsilons, originals + epsilons, perturbed
            )
            clipped_perturbed = ep.where(
                perturbation < -epsilons, originals - epsilons, clipped_perturbed
            )
            return clipped_perturbed
        else:
            return (1.0 - epsilons) * originals + epsilons * perturbed

def clear_c_pc(dims:ep.Tensor,C:ep.Tensor,P_C:ep.Tensor)->Tuple[ep.Tensor,ep.Tensor]:
    return ep.where(dims,ep.ones_like(C),C),ep.where(dims,ep.zeros_like(P_C),P_C)

class ArrayQueue:
    def __init__(self, maxlen: int, N: int):
        self.data = np.full((maxlen, N), np.nan, dtype=np.float32)
        self.next = 0
        self.tensor: Optional[ep.Tensor] = None

    @property
    def maxlen(self) -> int:
        return int(self.data.shape[0])

    @property
    def N(self) -> int:
        return int(self.data.shape[1])

    def append(self, x: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = x
        x = x.numpy()
        assert x.shape == (self.N,)
        self.data[self.next] = x
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = dims  # pragma: no cover
        dims = dims.numpy()
        assert dims.shape == (self.N,)
        assert dims.dtype == np.bool
        self.data[:, dims] = np.nan

    def mean(self) -> ep.Tensor:
        assert self.tensor is not None
        result = np.nanmean(self.data, axis=0)
        return ep.from_numpy(self.tensor, result)

    def isfull(self) -> ep.Tensor:
        assert self.tensor is not None
        result = ~np.isnan(self.data).any(axis=0)
        return ep.from_numpy(self.tensor, result)


def draw_proposals(
    bounds: Bounds,
    originals: ep.Tensor,
    adv: ep.Tensor,
    spherical_steps: ep.Tensor,
    source_steps: ep.Tensor,
    C: ep.Tensor,
    P_C: ep.Tensor,
    cc,
    cv,
    Orthogonal_setting
) -> Tuple[ep.Tensor, ep.Tensor, ep.Tensor, ep.Tensor]:
    # remember the actual shape
    shape = originals.shape
    ndim = originals.ndim
    bias = originals - adv + 1e-5
    bias_quant = ep.norms.l2(flatten(bias), axis=-1)
    bias_norm = bias / atleast_kd(bias_quant, ndim)

    # flatten everything to (batch, size)
    originals = flatten(originals)
    adv = flatten(adv)
    C = flatten(C)
    P_C = flatten(P_C)
    bias = flatten(bias)
    bias_norm = flatten(bias_norm)
    N, D = originals.shape

    base_tensor_type = adv
    # draw from an iid Gaussian (we can share this across the whole batch)
    x = ep.normal(base_tensor_type, (N, D))  # ∼ N(0, I)
    x *= C.sqrt()                     # ∼ N(0, C) 

    sigma = atleast_kd(spherical_steps * bias_quant, x.ndim)
    x =  sigma * (x/  atleast_kd(ep.norms.l2(x, axis=-1), x.ndim)) 

    if Orthogonal_setting:
        # rescale
        
        # make orthogonal (source_directions are normalized)
        y = x - atleast_kd((bias_norm * x).sum(axis=-1),x.ndim) * bias_norm  # Z 
        
        spherical_candidates = adv + y
    else:
        y  = x
        spherical_candidates = adv + x
        
        
     # clip
    min_, max_ = bounds
    spherical_candidates = spherical_candidates.clip(min_, max_)

    z = y + atleast_kd(source_steps, bias.ndim) * bias

    # # step towards the original inputs
    # new_source_directions = originals - spherical_candidates
    # candidates = spherical_candidates + atleast_kd(source_mu,new_source_directions.ndim) * new_source_directions

    candidates = adv + z
    # clip
    candidates = candidates.clip(min_, max_)

    # update the distribution
    # cc = 0.5
    # c1 = 0.1
    derta = (spherical_candidates - adv)/bias_norm*atleast_kd((spherical_steps.square() +source_steps.square()).sqrt(),candidates.ndim)
    # if Orthogonal_setting:
    #     derta = (spherical_candidates - adv)/bias_norm*atleast_kd((spherical_steps.square() +source_steps.square()).sqrt(),candidates.ndim)
    # else:
    #     derta = (spherical_candidates - adv)/bias_norm*atleast_kd((source_steps.square()).sqrt(),candidates.ndim)
    P_C = (1-cc)*P_C + (ep.ones_like(P_C)*cc*(2-cc)).sqrt() *derta
    C = (1-cv)*C + cv*P_C*P_C

    # restore shape
    candidates = candidates.reshape(shape)
    spherical_candidates = spherical_candidates.reshape(shape)
    C = C.reshape(shape)
    P_C = P_C.reshape(shape)
    return candidates, spherical_candidates, C, P_C
