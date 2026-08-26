"""HS-01 full analysis: tables -> analysis/outputs/hs01/, figures -> vault.

Run:  conda run -n uni python -m analysis.hs01.run_all
"""

from __future__ import annotations

import shutil

from . import figures, tables
from .load import analysis_frames


def main() -> None:
    print("loading sessions + pool ...")
    frames = analysis_frames()
    n = len(frames["sessions"])
    print(f"  {len(frames['sessions_all'])} sessions ({n} completed), "
          f"{len(frames['trials'])} analysis trials")

    print("tables ->", tables.OUT)
    tabs = tables.write_tables(frames)

    print("figures ...")
    assets = figures.generate_all(frames, tabs)
    print("figures ->", assets)

    # mirror the tables into the vault next to the figures
    data_dir = assets / "data"
    data_dir.mkdir(exist_ok=True)
    for f in tables.OUT.glob("*.csv"):
        shutil.copy2(f, data_dir / f.name)
    shutil.copy2(tables.OUT / "tables.md", data_dir / "tables.md")
    print("table copies ->", data_dir)


if __name__ == "__main__":
    main()
