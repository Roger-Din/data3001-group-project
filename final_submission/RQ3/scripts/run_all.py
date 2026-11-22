"""Run the complete RQ3 pipeline from the DATA3001 project root."""
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parents[2]
scripts = ["04_integrity_and_drifter_audit.py", "03_paired_cluster_bootstrap.py"]
for script in scripts:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / script)], cwd=project, check=True)
print("\nRQ3 pipeline completed. Review RQ3/output/ and the audit summary.")
