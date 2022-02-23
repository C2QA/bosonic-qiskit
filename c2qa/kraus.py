import math


import scipy


def calculate_kraus(num_photons: int, photon_loss_rate: float, time: float, a, a_dag):
    """
    Calculate Kraus operator given number of photons and photon loss rate over specified time. 

    Following equation 44 from Bosonic Oprations and Measurements, Girvin
    """
    n_hat = a * a_dag
    kraus = math.sqrt( math.pow( (1 - math.exp(-1 * photon_loss_rate * time)), num_photons ) / math.factorial(num_photons) )
    kraus = kraus * scipy.sparse.linalg.expm( -1 * (photon_loss_rate / 2) * time * n_hat )
    kraus = kraus * scipy.sparse.csr_matrix.power( a, num_photons )

    return kraus.toarray().tolist()
