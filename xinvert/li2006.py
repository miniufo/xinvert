"""Compatibility imports; implementation now lives in xinvert.helmholtz.

Existing ``helmholtz`` and ``reconstruct_flow`` calls retain their tuple return
contract. New code should use ``xinvert.helmholtz.decompose`` / ``reconstruct``.
"""

from .helmholtz._li2006 import helmholtz, reconstruct_flow

__all__ = ['helmholtz', 'reconstruct_flow']
