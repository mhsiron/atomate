# coding: utf-8



from __future__ import absolute_import, division, print_function, \
    unicode_literals

import warnings


"""
Defines standardized Fireworks that can be chained easily to perform various
sequences of VASP calculations.
"""

from fireworks import Firework

from atomate.vasp.firetasks.global_optimum_task import AnalyzeLossAndDecideNextStep


class GlobalOptimumFW(Firework):
    def __init__(self, structure, incar_grid, minimizer, previous_results=None,
                 max_fw=10, pmg_set=None, pmg_set_kwargs=None,
                 name ="Global Optimum", opt_kwargs=None, parents = None,
                 **kwargs):
        t = []
        t.append(AnalyzeLossAndDecideNextStep(structure, incar_grid,
                                              minimizer,
                                              previous_results=previous_results,
                                              max_fw=max_fw,
                                              pmg_set=pmg_set,
                                              pmg_set_kwargs=pmg_set_kwargs,
                                              opt_kwargs=opt_kwargs,
                                              parents=parents))

        super(GlobalOptimumFW,
              self).__init__(t, parents=parents,
                             name="{}-{}".format(
                                 structure.composition.reduced_formula, name),
                             **kwargs)

