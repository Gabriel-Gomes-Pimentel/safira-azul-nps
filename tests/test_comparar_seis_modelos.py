"""Contrato de desempate do comparativo; somente dados artificiais."""

import numpy as np
import pytest

from scripts.comparar_seis_modelos import selecionar_top_k


def test_top_k_preserva_ordem_dos_empates_e_capacidade():
    scores = np.array([0.2, 0.9, 0.9, 0.9, 0.1])

    assert selecionar_top_k(scores, 2).tolist() == [0, 1, 1, 0, 0]
    assert selecionar_top_k(scores, 0).sum() == 0
    assert selecionar_top_k(scores, len(scores)).sum() == len(scores)
    assert np.array_equal(selecionar_top_k(scores, 2), selecionar_top_k(scores, 2))


@pytest.mark.parametrize("scores,k", [([0.1], 2), ([0.1], -1), ([float('nan')], 1), ([[0.1]], 1)])
def test_top_k_recusa_entrada_invalida(scores, k):
    with pytest.raises(ValueError):
        selecionar_top_k(np.asarray(scores), k)
