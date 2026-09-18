"""A coherent TXID-correlated synthetic world, for controlled evaluation.

Why this exists
---------------
Elliptic++ gives a real chain layer with real labels and no network layer.
``network/synthetic.py`` gives a network layer whose announcements are
attached to Elliptic txids after the fact. The two are consistent in their
identifiers and in nothing else: no entity in the chain data *caused* an
announcement, so a question like "does a network-derived constraint prevent a
false merge" cannot be asked of them, only asserted about them.

This package generates both layers from ONE set of synthetic transactions. An
entity holds addresses, spends UTXOs, produces transactions with real input
and output values, and those same txids are then announced across the network
by that entity's origins. Chain structure and network structure share a
cause, so a controlled experiment over them measures something.

What it is not
--------------
It does not replace Elliptic++ and it does not replace the frozen network
dataset. It is a SEPARATE artifact tree with its own provenance, marked
SYNTHETIC_CONTROL, and no production endpoint serves it. A number measured
here is a property of a generator, not of Bitcoin, and every artifact says so.

The behaviours are not claims
-----------------------------
``PEELING``, ``MIXING_LIKE`` and the rest are transaction SHAPES chosen
because detectors should be able to recognise them. They are not claims about
how real offenders behave, and the adversarial half of the suite exists
precisely because several benign activities produce the same shapes.
"""

from __future__ import annotations

__all__ = ["behaviours", "generate", "overlap"]
