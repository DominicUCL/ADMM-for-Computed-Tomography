"""Alternating Direction method of Multipliers (ADMM) method variants."""

from __future__ import division
from builtins import range

import numpy
from odl.operator import Operator, OpDomainError, oputils, default_ops
import scipy

def admm_solver(x, f, g, L, tau, sigma, niter,data,space,solver,**kwargs):
    """    Parameters
    ----------
    x : ``L.domain`` element
        Starting point of the iteration, updated in-place.
    f, g : `Functional`
        The functions ``f`` and ``g`` in the problem definition. They
        need to implement the ``proximal`` method.
    L : linear `Operator`
        The linear operator that is composed with ``g`` in the problem
        definition. It must fulfill ``L.domain == f.domain`` and
        ``L.range == g.domain``.
    tau, sigma : positive float
        Step size parameters for the update of the variables.
    niter : non-negative int
        Number of iterations.
    solver: String
        Chose from "lsqr", "bicgstab" or "gmres

    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.

    """
    L_Operator= oputils.as_scipy_operator(L)
    Identity_matrix=oputils.as_scipy_operator(default_ops.IdentityOperator(space))
    A_Matrix = L_Operator.adjoint()*L_Operator+sigma*Identity_matrix
    callback = kwargs.pop('callback', None)
    tolerance = kwargs.pop('tolerance', 60e-5)

    z = space.zero()
    u = space.zero()
    data_array=data.asarray().ravel()
    prox_tau_f = f.proximal(tau)
    print(f"sigma: {sigma}")
    print(f"tau: {tau}")

    for _ in range(niter):
        
        B_Matrix = L_Operator.adjoint()*(data_array) + sigma*(z.asarray().ravel()+u.asarray().ravel()/sigma)

        if solver == "lsqr":
            results = scipy.sparse.linalg.lsqr(A_Matrix,B_Matrix,atol=tolerance, btol=tolerance)
        if solver == "bicgstab":
            results = scipy.sparse.linalg.bicgstab(A_Matrix,B_Matrix,tol=tolerance)
        if solver == "gmres":
            results = scipy.sparse.linalg.gmres(A_Matrix,B_Matrix,tol=tolerance)

        x[:] = results[0].reshape(128,128)
        prox_tau_f(x-u/tau,out=z)
        u[:] =  u - (x - z)
        if callback is not None:
            callback(x)
    return x
