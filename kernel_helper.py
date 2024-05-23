import numpy as np
import qml.kernels as k
from qml.math import cho_solve
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def fingerprint(smiles):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)

    fp_array = []
    for i in smiles:
        mol_i = Chem.MolFromSmiles(i)
        fp_i = fpgen.GetFingerprint(mol_i)
        fp_array.append(fp_i)

    return np.array(fp_array)

# Generate the kernel
def kernel(X1, X2=None, sigma=600):
    # if X1 is train
    if type(X2) == type(None):
        X2 = np.copy(X1)

    # Calculate the kernel function on the train X1
    K = k.matern_kernel(X1, X2, sigma, order=1, metric="l2")

    return K

def solve_alphas(X_train, lf_energy, delta_energy, lamda=0.001):

    kernel_array = kernel(X_train)

    kernel_array[np.diag_indices_from(kernel_array)] += lamda  # regularisation
    alpha = cho_solve(kernel_array, lf_energy)
    delta_alpha = cho_solve(kernel_array, delta_energy)

    return alpha, delta_alpha


def predict_from_alpha(X_train, X_test, alpha, delta_alpha):

    kernel_array = kernel(X_train, X_test)
    predicted_lf = np.dot(alpha, kernel_array)
    predicted_delta = np.dot(delta_alpha, kernel_array)

    return predicted_lf + predicted_delta, predicted_lf

def kernel_end2end(X_train, X_test, lf_energy, delta_energy):

    X_train = fingerprint(X_train)
    X_test = fingerprint(X_test)
    alpha, delta_alpha = solve_alphas(X_train,lf_energy,delta_energy)
    predicted_hf, predicted_lf = predict_from_alpha(X_train, X_test, alpha, delta_alpha)

    return predicted_hf, predicted_lf

