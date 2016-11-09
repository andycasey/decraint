

""" 
Experiment 1
------------

Use VDGMM to infer the number of clusters with 0% contamination, all common
abundances available, all available cluster members for N random clusters
(try from N = 1 to N = N_max).

In each Monte Carlo realisation, draw from the abundance uncertainties.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn import mixture
from collections import Counter

np.random.seed(123)

catalog = Table.read("data/catalog.fits")



# Which abundance columns should we use?
# Possible predictors:
# HE1 LI1 C1 C2 C3 C_C2 N2 N3 N_CN O1 O2 NE1 NE2 NA1 MG1 MG2 AL1 AL2 AL3 SI1 SI2
# SI3 SI4 S1 S2 S3 CA1 CA2 SC1 SC2 TI1 TI2 V1 V2 CR1 CR2 MN1 FE1 FE2 FE3 CO1 NI1 
# CU1 ZN1 SR1 Y1 Y2 ZR1 ZR2 NB1 MO1 RU1 BA2 LA2 CE2 PR2 ND2 SM2 EU2 GD2 DY2

# These abundances are the ones that are most commonly available.
# There are 2961 stars with all of these abundances.
# Of them, 356 are cluster members and the rest are field stars.

predictors = ("RA", "DEC", "VEL")

#("FE1", "MG1", "TI1", "SI1", "CA1", "AL1", "CR1", "MN1", "NI1", "CO1", "BA2")

# Exclude stars that do not have these predictors, or ones that are in the field
keep = catalog["group"] > -1
for predictor in predictors:
    keep *= np.isfinite(catalog[predictor]) #* np.isfinite(catalog["E_" + predictor])
catalog = catalog[keep]

# Number of Monte-Carlo realisations to do for each number of true clusters N
M = 100

# Number of potential clusters.
N = len(set(catalog["group"])) 


for prior in (1e-4, 1e-6, 1e-8, 1e-2, 1, 10, 100, 10000, 1e5, 1e6, 1e7):

    realisations = {}
    for n in range(10, N + 1):

        print("Running experiment with N = {}".format(n))

        realisations[n] = []
        for m in range(M):

            # Get cluster indices
            cluster_indices = np.random.choice(range(N), size=n, replace=False)
            match = np.in1d(catalog["group"], cluster_indices)

            # Draw from the abundances for the same number of stars in those clusters
            #K = max(Counter(catalog["group"][match]).values())

            """
            X = np.array([
                np.random.normal(
                    loc=catalog[predictor][match], 
                    scale=catalog["E_" + predictor][match]) \
                for predictor in predictors
                ])
            """
            X = np.array([catalog[predictor][match] for predictor in predictors])

            # Use a VDPGMM to infer the number of clusters.

            # Construct the model, allowing for up to twice as many clusters.
            n_components = min(2 * N, X.shape[1])
            dpgmm = mixture.BayesianGaussianMixture(
                n_components=n_components, n_init=1, max_iter=10000,
                covariance_type="full", weight_concentration_prior=prior)
            fit = dpgmm.fit(X.T)

            Y = fit.predict(X.T)
            n_inferred = len(set(Y))
        
            # Record the number of inferred clusters.
            realisations[n].append(n_inferred)
            print(n, n_inferred)

    x = np.array(realisations.keys())
    y = np.array([realisations[k] for k in x])


    fig, ax = plt.subplots()
    ax.scatter(x, np.median(y, axis=1), facecolor="b")
    ax.fill_between(x, np.percentile(y, 16, axis=1), np.percentile(y, 84, axis=1),
        facecolor="b", alpha=0.5, zorder=-1)

    ax.set_xlim(0, 1 + max(x))
    ax.set_ylim(0, 1 + max(x))

    ax.set_title(prior)

    fig.savefig("scripts/hyper-parameter-search-full-{}.png".format(prior))


raise a

