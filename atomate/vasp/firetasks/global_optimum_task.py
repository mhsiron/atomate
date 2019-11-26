from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

from fireworks import FiretaskBase, FWAction, explicit_serialize, Workflow

import numpy as np
from skopt import gp_minimize, gbrt_minimize, forest_minimize
from custodian.vasp.handlers import WalltimeHandler

from pymatgen import Structure
from pymatgen.io.vasp.sets import MPRelaxSet


from pymatgen.io.vasp.outputs import Oszicar

@explicit_serialize
class CalculateLoss(FiretaskBase):
    """
    doc string
    """
    required_params = ["current_incar_params"]

    def run_task(self, fw_spec):
        current_incar_params = self.get("current_incar_params")
        loss = loss_function()
        params_tried = self.get("params_tried")

        return FWAction(update_spec={"_push":
                                         {
                                             "results":
                                                 {current_incar_params:loss}
                                         }})


@explicit_serialize
class AnalyzeLossAndDecideNextStep(FiretaskBase):
    """
    Doc string
    """
    required_params = ["structure","incar_grid","minimizer","previous_results",
                       "max_fw","pmg_set","pmg_set_kwargs",
                       "opt_kwargs","parents"]

    def run_task(self, fw_spec):
        structure = self.get("structure")
        incar_grid = self.get("incar_grid")
        minimizer = self.get("minimizer","forest")
        if minimizer == "forest":
            minimizer = forest_minimize
        elif minimizer == "grbt":
            minimizer = gbrt_minimize
        elif minimizer == "gp":
            minimizer = gp_minimize
        previous_results = self.get("previous_results", None)
        max_fw = self.get("max_fw",10)
        pmg_set = self.get("pmg_set", None)
        pmg_set_kwargs = self.get("pmg_set_kwargs", None)
        opt_kwargs = self.get("opt_kwargs", None)
        parents = self.get("parents", None)

        if parents is not None:
            previous_results = fw_spec.get("results")
        wf = load_and_launch(structure=structure,
                             incar_grid=incar_grid,minimizer=minimizer,
                             previous_results=previous_results,
                             max_fw=max_fw, pmg_set=pmg_set,
                             pmg_set_kwargs=pmg_set_kwargs,
                             opt_kwargs=opt_kwargs)
        return FWAction(additions=wf)


def loss_function(metric="total_N", oszicar_file="OSZICAR.gz"):
    """
    Loss function used to determine optimal INCAR configuration,
    the default is the negative of total ionic steps run,
    :param metric:
    :return:
    """
    metrics = {}
    osz = Oszicar(oszicar_file)
    # total electronic step:
    last_elec_step = osz.electronic_steps[-1][-1]
    N = last_elec_step["N"] + (len(osz.electronic_steps) - 1) * len(
        osz.electronic_steps[0])
    metrics["total_N"] = -N  # the more total steps, the better, neg.

    # average dE of last 5 electronic steps
    if len(osz.electronic_steps[-1]) > 5:
        last_five_dE = np.average(
            [k["dE"] for k in osz.electronic_steps[-1]][-5:])
    else:
        last_five_dE = np.average([k["dE"] for k in osz.electronic_steps[-1]])
    metrics[
        "last_five_dE"] = last_five_dE  # the smaller the dE the better, pos

    # ionic step:
    num_ionic = len(osz.electronic_steps)
    metrics["num_ionic"] = -num_ionic  # the more ionic steps, the better, neg

    # dE of ionic step
    ionic_dE = osz.ionic_steps[-1]["dE"]
    metrics["ionic_dE"] = ionic_dE  # the smaller the ionic dE, the better, pos

    return metrics[metric]

def load_and_launch(structure, incar_grid, minimizer,
                previous_results=None,
                max_fw=10, pmg_set=None, pmg_set_kwargs=None,
                opt_kwargs=None):
    """
    Takes various incar parameters values to try out, a minimizer algorithm, and
    previous results from past calculations to figure out optimum INCAR params for convergence
    """
    pmg_set_kwargs = pmg_set_kwargs or None
    wall_time = int(60 * 60 * 1)  # 1 hour
    opt_kwargs = opt_kwargs or {}
    pmg_set_kwargs = pmg_set_kwargs or {}

    p_t = []  # INCAR params to try out, reformatted for skopt library
    for key, item in incar_grid.items():
        p_t.append(item)
    previous_results = previous_results or {}
    n_calls = 100
    # We want to limit the amount of fireworks that gets launched while
    # we are building our database of values. If the current database of
    # values is below the default n_calls, we set the maximum number of
    # initial calculations:
    if len(previous_results) < 100:
        n_calls = max_fw

    l_params = []  # This is the dictionary of params we have
    # attempted for these calculations so far.
    fws = []

    def func(params):
        print(params)
        if params in list(previous_results.keys()):
            # If we have already successfully run this calculation,
            # and we have supplied it,
            # we simply return the results of this calculation
            return previous_results[params]
        else:
            if params not in l_params:
                # This is where can launch the FW, with the incar
                # params supplied.

                # Update incar dictionary
                incar_update = {}
                for n, (key, item) in enumerate(incar_grid.items()):
                    incar_update[key] = params[n]

                # Create FW:
                # appropriate MP Set
                set = MPRelaxSet(structure, **pmg_set_kwargs) or \
                      pmg_set(structure, **pmg_set_kwargs)
                from atomate.vasp.fireworks.core import OptimizeFW
                fws.append(OptimizeFW(structure,
                                      vasp_input_set=set,
                                      vasp_cmd=">>vasp_cmd<<",
                                      db_file=">>db_file<<",
                                      job_type="normal",
                                      run_vasp_kwargs={
                                          "handler_group":
                                              [WalltimeHandler(
                                                  wall_time=wall_time)]
                                      },
                                      vasp_to_db_kwargs={
                                          "defuse_unsuccessful": False
                                      }, **opt_kwargs))

                # Append FireTask to pass proper data from OSZICAR:
                loss_task = CalculateLoss(current_incar_params=params)
                fws[-1].tasks.append(loss_task)

                from atomate.vasp.fireworks.global_optimum import \
                    GlobalOptimumFW
                # Add Fireworks to pass analyze data, and rerun function
                fws.append(GlobalOptimumFW(structure,incar_grid,minimizer,
                                           max_fw=max_fw,pmg_set=pmg_set,
                                           pmg_set_kwargs=pmg_set_kwargs,
                                           opt_kwargs=opt_kwargs,
                                           parents = fws))

                # For now return a random value
                return 0
            else:
                # We've already added this FW to run. For now we will
                # return a random value.
                return 0
            l_params.append(params)

    wf = Workflow(fws)
    minimizer(func, p_t, n_calls=n_calls)
    return wf