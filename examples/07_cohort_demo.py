import numpy as np
import matplotlib.pyplot as plt
from pydlvh import DLVH, DLVHCohort

# ---------- synthetic patient generator ----------
def make_patient(n_voxels=4000,
                 dose_max=60.0,
                 mu_dose=30.0, sigma_dose=7.0,
                 mu_let=3.0,  sigma_let=1.0,
                 volume_rng=(80.0, 120.0)):
    """
    Crea un paziente sintetico:
    - dose uniforme in [0, dose_max]
    - LET normale troncata a >=0
    - pesi relw ~ exp(-(dose-mu)^2/(2*sigma^2)) per modellare DVH diff ~ gaussiana
    """
    dose = np.random.uniform(0.0, dose_max, size=n_voxels)
    let  = np.random.normal(loc=mu_let, scale=sigma_let, size=n_voxels)
    let  = np.clip(let, 0.0, None)

    relw_raw = np.exp(-0.5 * ((dose - mu_dose) / max(sigma_dose, 1e-6))**2)
    # evita tutti zero (patologico ma per completezza)
    if not np.any(relw_raw > 0):
        relw_raw[:] = 1.0
    relw = relw_raw / relw_raw.sum()

    volume_cc = np.random.uniform(*volume_rng)
    return DLVH(dose=dose, let=let, volume_cc=volume_cc, relative_volumes=relw)


def main():
    np.random.seed(7)

    # ----- costruisci 5 pazienti con parametri leggermente diversi -----
    dose_shapes = [(27, 6.0), (30, 7.0), (33, 8.0), (29, 6.5), (32, 7.5)]
    dlvhs = [make_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes]
    cohort = DLVHCohort(dlvhs)

    # ===================== 1) DVH MEDIAN + IQR =====================
    dvh_med = cohort.aggregate_1d(
        quantity="dose",
        stat="median",
        normalize=True,
        cumulative=True  # DVH cumulativa classica
    )
    ax = dvh_med.plot(color="C0", label="Median DVH", show_band=True)
    ax.legend(loc="best")
    ax.set_title("Cohort DVH (median) with IQR band")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ===================== 2) DVH MEAN ± STD =====================
    dvh_mean = cohort.aggregate_1d(
        quantity="dose",
        stat="mean",
        normalize=True,
        cumulative=True
    )
    ax = dvh_mean.plot(color="C1", label="Mean DVH", show_band=True)
    ax.legend(loc="best")
    ax.set_title("Cohort DVH (mean) with ±std band")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (opzionale) LVH mediano con IQR
    lvh_med = cohort.aggregate_1d(
        quantity="let",
        stat="median",
        normalize=False,  # LVH spesso ha senso in cm^3
        cumulative=True
    )
    ax = lvh_med.plot(color="C2", label="Median LVH", show_band=True)
    ax.legend(loc="best")
    ax.set_title("Cohort LVH (median) with IQR band")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ===================== 3) DLVH 2D MEDIAN + P25/P75 =====================
    h2d_med = cohort.aggregate_2d(
        stat="median",
        normalize=True,
        cumulative=True
    )
    # mappa centrale (mediana)
    h2d_med.plot(cmap="plasma", mode="values", isovolumes=[5, 10, 20])
    plt.title("Cohort DLVH 2D (median)")
    plt.tight_layout()
    plt.show()

    # mappa p25
    h2d_med.plot(cmap="viridis", mode="p_lo")
    plt.title("Cohort DLVH 2D - 25th percentile (IQR lower)")
    plt.tight_layout()
    plt.show()

    # mappa p75
    h2d_med.plot(cmap="viridis", mode="p_hi")
    plt.title("Cohort DLVH 2D - 75th percentile (IQR upper)")
    plt.tight_layout()
    plt.show()

    # (facoltativo) estrai i margini dal 2D aggregato (es. DVH mediano dai margini)
    dvh_from2d = cohort.aggregate_marginals(kind="dose", stat="median",
                                            normalize=True, cumulative=True)
    ax = dvh_from2d.plot(color="C3", label="DVH from 2D median", show_band=True)
    ax.legend(loc="best")
    ax.set_title("DVH marginal from 2D (median + IQR)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
