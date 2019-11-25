import numpy as np
from skopt import gp_minimize, gbrt_minimize, forest_minimize
from atomate.vasp.fireworks import OptimizeFW
from pymatgen.io.vasp.sets import MPRelaxSet
from custodian.vasp.handlers import WalltimeHandler
from atomate.vasp.firetasks.global_optimum_task import CalculateLoss


def get_optimum_incar_parameters_wf():
    pass



