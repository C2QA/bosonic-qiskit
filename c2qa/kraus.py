import math


import c2qa
import scipy


def calculate_kraus(num_photons: int, photon_loss_rate: float, time: float, ops: c2qa.operators.CVOperators):
    """
    Calculate Kraus operator given number of photons and photon loss rate over specified time. 

    Following equation 44 from Bosonic Oprations and Measurements, Girvin
    """
    operators = []

    for photons in range(num_photons + 1):
        kraus = math.sqrt( math.pow( (1 - math.exp(-1 * photon_loss_rate * time)), photons ) / math.factorial(photons) )
        kraus = kraus * scipy.sparse.linalg.expm( -1 * (photon_loss_rate / 2) * time * ops.N )
        kraus = kraus.dot( ops.a ** photons )
        operators.append(kraus.toarray())

    return operators
