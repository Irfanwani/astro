# %%
# import numpy, astropy and matplotlib for basic functionalities
from utils import vl, vsp, vsscp
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pkg_resources

# import agnpy classes
from agnpy.spectra import BrokenPowerLaw
from agnpy.fit import SynchrotronSelfComptonModel, load_gammapy_flux_points
from agnpy.utils.plot import load_mpl_rc, sed_y_label

load_mpl_rc()

# import gammapy classes


print(vl(0.1))

print("{:.2e}".format(vsp(20, 1, 1000, vl(0.1))))

print(vsscp(20, 1, 1000, vl(0.1)))


# %%
# electron energy distribution
n_e = BrokenPowerLaw(
    k=1e-8 * u.Unit("cm-3"),
    p1=2.02,
    p2=3.43,
    gamma_b=1000,  # 1e5
    gamma_min=500,
    gamma_max=1e6,
)

# initialise the Gammapy SpectralModel
ssc_model = SynchrotronSelfComptonModel(n_e, backend="gammapy")

# %%
ssc_model.parameters["z"].value = 0.5  # 0.0308
ssc_model.parameters["delta_D"].value = 10  # 18 b=0.1
ssc_model.parameters["t_var"].value = (1 * u.d).to_value("s")
ssc_model.parameters["t_var"].frozen = True
ssc_model.parameters["log10_B"].value = -1  # -1.3

# %%
ssc_model.parameters.to_table()

# %%
ssc_model.spectral_parameters.to_table()


# %%
ssc_model.emission_region_parameters.to_table()


# %%
sed_path = pkg_resources.resource_filename(
    "agnpy", "data/mwl_seds/Mrk421_2011.ecsv")

systematics_dict = {
    "Fermi": 0.10,
    "GASP": 0.05,
    "GRT": 0.05,
    "MAGIC": 0.30,
    "MITSuME": 0.05,
    "Medicina": 0.05,
    "Metsahovi": 0.05,
    "NewMexicoSkies": 0.05,
    "Noto": 0.05,
    "OAGH": 0.05,
    "OVRO": 0.05,
    "RATAN": 0.05,
    "ROVOR": 0.05,
    "RXTE/PCA": 0.10,
    "SMA": 0.05,
    "Swift/BAT": 0.10,
    "Swift/UVOT": 0.05,
    "Swift/XRT": 0.10,
    "VLBA(BK150)": 0.05,
    "VLBA(BP143)": 0.05,
    "VLBA(MOJAVE)": 0.05,
    "VLBA_core(BP143)": 0.05,
    "VLBA_core(MOJAVE)": 0.05,
    "WIRO": 0.05,
}

# define minimum and maximum energy to be used in the fit
E_min = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
E_max = 100 * u.TeV

datasets = load_gammapy_flux_points(sed_path, E_min, E_max, systematics_dict)

# %%
sky_model = SkyModel(spectral_model=ssc_model, name="Mrk421")
datasets.models = [sky_model]

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for dataset in datasets:
    dataset.data.plot(ax=ax, label=dataset.name)


xdata = list(np.concatenate([i.get_xdata() for i in ax.get_lines()]))


ydata = [i.get_ydata() for i in ax.get_lines()]


resultdata = list(np.concatenate(ydata))


maxval = max(resultdata)

print(maxval)

xval = xdata[resultdata.index(maxval)]
print(xval)


ssc_model.plot(
    ax=ax,
    energy_bounds=[1e-6, 1e14] * u.eV,
    energy_power=2,
    label="SSC model",
    color="k",
    lw=1.6,
)

ax.set_ylabel(sed_y_label)
ax.set_xlabel(r"$E\,/\,{\rm eV}$")
ax.set_xlim([1e-6, 1e14])
ax.legend(ncol=4, fontsize=9)

ax.annotate('Maxima', (xval, maxval), xytext=(9000, 4.10603e-10),
            arrowprops={'facecolor': 'orange', 'arrowstyle': 'fancy', 'linewidth': 0.5})

plt.show()

# %%

# define the fitter
fitter = Fit()
results = fitter.run(datasets)

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for dataset in datasets:
    dataset.data.plot(ax=ax, label=dataset.name)

ssc_model.plot(
    ax=ax,
    energy_bounds=[1e-6, 1e14] * u.eV,
    energy_power=2,
    label="model",
    color="k",
    lw=1.6,
)

# plot a line marking the minimum energy considered in the fit
ax.axvline(E_min, ls="--", color="gray")

plt.legend(ncol=4, fontsize=9)
plt.xlim([1e-6, 1e14])
plt.show()

# %%
# plot the covariance matrix
ssc_model.covariance.plot_correlation()
plt.grid(False)
plt.show()

# %%
# plot the profile for the normalisation of the electron energy distribution
par = sky_model.spectral_model.log10_k
par.scan_n_values = 50
profile = fitter.stat_profile(datasets=datasets, parameter=par)

print(par.name)

print(profile)

# to compute the delta TS
total_stat = results.total_stat
# profile["log10_k_scan"]
plt.plot(profile[f"Mrk421.spectral.{par.name}_scan"],
         profile["stat_scan"] - total_stat)
plt.ylabel(r"$\Delta(TS)$", size=12)
plt.xlabel(r"$log_{10}(k_{\rm e} / {\rm cm}^{-3})$", size=12)
plt.show()
