
from pathlib import Path
import copernicusmarine

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "external" / "glorys"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1M-m"

for year in range(1998, 2023):
    filename = f"glorys_surface_{year}.nc"
    output_file = OUTPUT_DIR / filename

    if output_file.exists():
        print(f"Skipping {year}: file already exists")
        continue

    print(f"\nDownloading GLORYS {year}...")

    try:
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=["uo", "vo"],
            minimum_longitude=124,
            maximum_longitude=136,
            minimum_latitude=24,
            maximum_latitude=36,
            minimum_depth=0.49402499,
            maximum_depth=0.5,
            start_datetime=f"{year}-01-01T00:00:00",
            end_datetime=f"{year}-12-31T23:59:59",
            output_directory=str(OUTPUT_DIR),
            output_filename=filename,
        )

        print(f"Finished {year}")

    except Exception as e:
        print(f"FAILED {year}: {e}")
        print("Stopping so the issue can be checked.")
        break

print("\nDownload process finished.")
