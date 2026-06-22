"""Objective-driven membership-function auto-tuning (self-designed ablation).

Motivation. The baseline ANFIS uses *hand-set* Gaussian membership functions
(the expert ``LINGUISTIC_TERMS``). The earlier rec2 study tried to *replace* them
with data-clustered centres and failed, because k-means optimises data density,
not the control objective. This study instead keeps the linguistic partition
(term names/order, so the rule base stays valid) and lets a derivative-free
optimiser place the Gaussian centres/widths to directly minimise closed-loop MRT
on a calibration set -- seeded at the expert design and elitist, so it can only
match or beat the hand-tuned baseline on the calibration scenarios ("without
harming performance").
"""
