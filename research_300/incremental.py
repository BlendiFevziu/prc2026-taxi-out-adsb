"""Statistics for comparing two detector arms against production-strength predictions. Pure functions, no I/O.

A policy predicts the detector's taxi time where its arm gives an estimate and the comparator's prediction
elsewhere. The incremental value of arm B over arm A is, per departure, the squared error of A's policy minus that
of B's (positive = B better). Rows where both policies predict the same contribute zero, so the mean over ALL
departures is the effect per departure and scales directly to the official set by the airport's share.
"""
import numpy as np


def policy(truth_free_pred, detector_taxi):
    """Detector taxi time where available (finite), else the comparator's prediction."""
    d = np.asarray(detector_taxi, float)
    return np.where(np.isfinite(d), d, np.asarray(truth_free_pred, float))


def gain_rows(y, comparator, taxi_a, taxi_b):
    y = np.asarray(y, float)
    a, b = policy(comparator, taxi_a), policy(comparator, taxi_b)
    return (a - y) ** 2 - (b - y) ** 2


def without_top(gain, share=0.05):
    """Mean gain over the rows that CHANGED, after removing the most favourable `share` of them."""
    g = np.asarray(gain, float)
    g = g[g != 0]
    if not len(g):
        return 0.0
    keep = np.ones(len(g), bool)
    keep[np.argsort(g)[::-1][:int(np.ceil(share * len(g)))]] = False
    return float(g[keep].mean())


def block_bootstrap(gain, blocks, draws=2000, seed=0):
    """95 % interval of the mean gain per departure, resampling whole blocks (airport-days) with replacement."""
    gain, blocks = np.asarray(gain, float), np.asarray(blocks)
    keys = np.unique(blocks)
    sums = np.array([gain[blocks == k].sum() for k in keys])
    counts = np.array([(blocks == k).sum() for k in keys])
    pick = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
