
from pathlib import Path
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
GLORYS_DIR = ROOT / "data" / "external" / "glorys"

files = sorted(GLORYS_DIR.glob("*.nc"))

print("=== GLORYS FILE CHECK ===")
print("Number of files:", len(files))

for file in files:
    print("\n----------------------------")
    print("File:", file.name)

    try:
        with xr.open_dataset(file, engine="h5netcdf") as ds:
            print("Dimensions:", dict(ds.sizes))
            print("First month:", str(ds.time.values[0]))
            print("Last month:", str(ds.time.values[-1]))
            print("Number of months:", ds.sizes["time"])
            print("Variables:", list(ds.data_vars))

    except Exception as e:
        print("ERROR:", e)
