import numpy as np

# eps_r: relative permittivity (real part), used at mm-wave frequencies.
# Values follow ITU-R P.2040 (building materials table). Verify against the
# recommendation before citing them in the paper.
MATERIALS = {
    "concrete":     {"eps_r": 5.31, "source": "ITU-R P.2040"},
    "wood":         {"eps_r": 1.99, "source": "ITU-R P.2040"},
    "glass":        {"eps_r": 6.27, "source": "ITU-R P.2040"},
    "plasterboard": {"eps_r": 2.94, "source": "ITU-R P.2040"},
    "metal":        {"eps_r": None, "source": "perfect conductor assumption"},
}


# Human body (not a room material, so it is NOT in MATERIALS: the scenario
# file cannot use it for furniture). Complex permittivity eps' - j*eps''.
# Skin at 60 GHz, Gabriel model: 8.0 - j10.9 (Table I of Wu, Rappaport and
# Collins, IEEE ICC 2015, arXiv:1503.05944). The same paper reports that
# 34-42% of the normally incident power is reflected by skin at 60 GHz.
# Assumptions: bare skin only (no clothing), 60 GHz (revisit when the radar
# config of step 25 fixes f0), no angle dependence.
BODY_MATERIALS = {
    "skin": {"eps_r": complex(8.0, -10.9), "source": "Gabriel, via Wu et al. 2015"},
}


def fresnel_power(eps):
    """Power reflectance at normal incidence for a real or complex eps_r."""
    n = np.sqrt(complex(eps))
    gamma = (1.0 - n) / (1.0 + n)
    return float(abs(gamma) ** 2)


def reflectance(name):
    """Power reflectance at normal incidence (Fresnel), lossless approximation."""
    eps = MATERIALS[name]["eps_r"]
    if eps is None:
        return 1.0
    return fresnel_power(eps)


def body_reflectance(name="skin"):
    """Power reflectance of a body material (complex permittivity)."""
    return fresnel_power(BODY_MATERIALS[name]["eps_r"])


def apply_materials(room):
    """Add 'reflectance' and 'strength' (= area * reflectance) per point."""
    table = {m: reflectance(m) for m in np.unique(room["material"])}
    refl = np.array([table[m] for m in room["material"]])
    out = dict(room)
    out["reflectance"] = refl
    out["strength"] = room["area"] * refl
    return out