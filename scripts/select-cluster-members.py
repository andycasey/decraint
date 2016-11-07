#!/usr/bin/env python

""" Create realisations of cluster and field stars. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"


import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from itertools import cycle
from sklearn import mixture


# Load in the data.
data = Table.read("data/GES_iDR4_WG15_Recommended_Abundances_20042016.fits")

# Select cluster members based on constraints.
with open("data/clusters.yaml", "r") as fp:
    cluster_rules = yaml.load(fp)

# Specify a MEMBERSHIP column for each row, which defaults to 'FIELD'
default_membership = "FIELD  "
data["MEMBERSHIP"] = [default_membership] * len(data)
trimmed_ges_flds = np.array([each.strip() for each in data["GES_FLD"]])

for ges_fld, rules in cluster_rules.items():

    match = (trimmed_ges_flds == ges_fld)
    assert any(match), "No stars matching the GES_FLD '{}'".format(ges_fld)

    for key, constraint in rules.items():
        assert key in data.dtype.names, "Is this a custom functional constraint?"

        lower, upper = (min(constraint), max(constraint))
        match *= (upper >= data[key]) * (data[key] >= lower)

    # Ensure that stars cannot be matched to more than one cluster.
    assert len(set(data["MEMBERSHIP"][match])) == 1 \
    and data["MEMBERSHIP"][match][0].strip() == default_membership.strip(), \
        "Star cannot be assigned to multiple clusters."

    data["MEMBERSHIP"][match] = ges_fld

    assert match.sum() > 0, "No stars matched to {}".format(ges_fld)
    print(ges_fld, match.sum())

# Some requirements for usefulness:
feature_columns = ("MG1", "TI2", "FEH")
for feature_column in feature_columns:
    data["MEMBERSHIP"][~np.isfinite(data[feature_column])] = default_membership

is_cluster_member = np.array([each != default_membership for each in data["MEMBERSHIP"]])
N_cluster_members = sum(is_cluster_member)

cluster_names = list(set(data["MEMBERSHIP"][is_cluster_member]).difference([default_membership]))
N_clusters = len(cluster_names)


# Do a GMM with all cluster members to make sure we can identify them.
X = np.array([data[feature_column][is_cluster_member] \
    for feature_column in feature_columns]).T

dpgmm = mixture.BayesianGaussianMixture(
    n_components=2 * N_clusters, n_init=1, max_iter=10000).fit(X)

# Plot the results.
Y_ = dpgmm.predict(X)

fig, ax = plt.subplots()
colors = cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

for i, (mean, cov, color) in enumerate(zip(dpgmm.means_, dpgmm.covariances_, colors)):

    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0 * v)
    u = w[0] / np.linalg.norm(w[0])

    match = Y_ == i
    if not np.any(match):
        continue

    ax.scatter(X[match, 0], X[match, 1], 0.8, color=color)

    angle = (180.0 / np.pi) * (np.arctan(u[1] / u[0]))
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)

