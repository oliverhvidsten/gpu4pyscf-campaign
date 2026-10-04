"""jobflow/FireWorks campaign driver for the qcflow-gpu4pyscf GPU geometry optimizer.

Pipeline (see README):  structures JSON -> (optional) Lewis starting geometries
-> one Firework per molecule in a FireWorks launchpad -> ``rlaunch rapidfire``
per GPU on the cluster -> task documents in a jobflow JobStore (MongoDB + GridFS).

Nothing in this package imports cupy/gpu4pyscf at module level: the loader runs
on login nodes, and inside a job the CUDA device must be chosen
(``CUDA_VISIBLE_DEVICES``) before ``crn_utils.pyscf_calc`` is imported.
"""

__version__ = "0.1.0"
