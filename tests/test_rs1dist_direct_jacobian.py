import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from diffcollision.core.rs1dist import (  # noqa: E402
    _normalize_vector_jacobian,
    _rs1dist_wp_jacobian,
)
from diffcollision.utils import torch_normalize_vector  # noqa: E402


def _rs1dist_wp_func(Ti, wpj, lsi_o):
    lsi = lsi_o[:, :3] @ Ti[:3, :3].T + Ti[:3, 3]
    pdist = (lsi - wpj).norm(dim=-1)
    weight = torch.softmax(-pdist / pdist.std().sqrt(), dim=-1)
    wpi = (weight.unsqueeze(-1) * lsi).sum(dim=-2)
    lti = lsi_o[:, 3:] @ Ti[:3, :3].T
    ni = (weight.unsqueeze(-1) * lti).sum(dim=-2)
    return (wpi, ni), ni


def _autograd_rs1dist_wp_jacobian(Ti, wpj, lsi_o):
    jacb_fun = torch.vmap(
        torch.func.jacrev(_rs1dist_wp_func, argnums=(0, 1), has_aux=True)
    )
    ((J_wp_T, J_wp_wpj), (J_n_T, J_n_wpj)), ni = jacb_fun(Ti, wpj, lsi_o)
    return J_wp_T, J_wp_wpj, J_n_T, J_n_wpj, ni


def test_direct_rs1dist_wp_jacobian_matches_autograd_reference():
    torch.manual_seed(7)
    n_contact = 5
    n_local = 8
    Ti = torch.randn(n_contact, 4, 4, dtype=torch.double)
    Ti[:, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.double)
    wpj = torch.randn(n_contact, 3, dtype=torch.double)
    lsi_o = torch.randn(n_contact, n_local, 6, dtype=torch.double)

    direct = _rs1dist_wp_jacobian(Ti, wpj, lsi_o)
    autograd = _autograd_rs1dist_wp_jacobian(Ti, wpj, lsi_o)

    for direct_value, autograd_value in zip(direct, autograd):
        torch.testing.assert_close(
            direct_value,
            autograd_value,
            rtol=1e-9,
            atol=1e-10,
        )


def test_direct_normal_jacobian_matches_autograd_reference():
    torch.manual_seed(11)
    n1 = torch.randn(6, 3, dtype=torch.double)
    n2 = torch.randn(6, 3, dtype=torch.double)
    n2[0] = n1[0]

    def normal_func(n1i, n2i):
        return torch_normalize_vector(n1i - n2i)

    jacb_fun = torch.vmap(torch.func.jacrev(normal_func, argnums=(0, 1)))
    J_n_n1, J_n_n2 = jacb_fun(n1, n2)
    direct = _normalize_vector_jacobian(n1 - n2)

    torch.testing.assert_close(direct, J_n_n1, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(-direct, J_n_n2, rtol=1e-9, atol=1e-10)
