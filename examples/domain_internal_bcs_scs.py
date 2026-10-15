"""Plot domain topology and internal boundaries around the South China Sea."""
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import ultraplot as uplt
import xarray as xr
from matplotlib.colors import ListedColormap

from xinvert.domain import analyze_domain
from xinvert.internal_bcs import identify_internal_bcs

DATA_PATH = Path("Data/SODA.nc")
OUTPUT_PATH = Path("docs/source/_static/domain_internal_bcs_south_china_sea.png")
REGION = {"lon": slice(98, 125), "lat": slice(0, 25)}


with xr.open_dataset(DATA_PATH) as dataset:
    taux = dataset["taux"].isel(time=0).load()

# Analyse the global grid first.  Cropping before analysis would make the crop
# edges external boundaries and could change connected-component labels.
topology = analyze_domain(
    taux,
    dims=("lat", "lon"),
    periodic="lon",
)
internal_bcs = identify_internal_bcs(topology)

field = taux.sel(**REGION)
active = topology.active.sel(**REGION)
labels = topology.component_labels.where(topology.active).sel(**REGION)
face_count = sum(
    internal_bcs.faces[direction].astype(np.int8)
    for direction in ("east", "west", "north", "south")
).sel(**REGION)

projection = ccrs.Mercator(central_longitude=111.5)
transform = ccrs.PlateCarree()
fig, axes = uplt.subplots(
    ncols=3, proj=projection, refwidth=4.1, share=False,
)

limit = float(np.nanpercentile(np.abs(field.values), 99))
field.plot.pcolormesh(
    ax=axes[0], x="lon", y="lat", transform=transform,
    cmap="RdBu_r", vmin=-limit, vmax=limit,
    add_labels=False, cbar_kwargs={"label": r"$\tau_x$"},
)

active.where(active).astype(float).plot.pcolormesh(
    ax=axes[1], x="lon", y="lat", transform=transform,
    cmap=ListedColormap(["#2166ac"]),
    add_colorbar=False, add_labels=False,
)

face_count.where(face_count > 0).plot.pcolormesh(
    ax=axes[2], x="lon", y="lat", transform=transform,
    cmap="magma", vmin=1, vmax=4, add_labels=False,
    cbar_kwargs={"label": "internal faces per wet cell"},
)

axes.format(
    abc="(a)",
    coast=True,
    reso="hi",
    land=False,
    borders=False,
    rivers=False,
    lakes=False,
    grid=False,
    labels=False,
    lonlim=(98, 125),
    latlim=(0, 25),
)
axes[0].format(title="SODA zonal wind stress")
axes[1].format(title="Active ocean domain")
axes[2].format(title="Homogeneous-Neumann internal faces")
fig.suptitle(
    "xinvert domain and internal-boundary discovery: South China Sea",
    fontsize=14,
)
fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")

print(f"global_components={topology.n_components}")
print(f"regional_active_cells={int(active.sum())}")
print(f"regional_component_labels={np.unique(labels.values[np.isfinite(labels)])}")
print(f"regional_directed_internal_faces={int(face_count.sum())}")
print(f"figure={OUTPUT_PATH}")
