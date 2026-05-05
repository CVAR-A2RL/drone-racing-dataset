import argparse
import os
from glob import glob

import numpy as np
import pandas as pd


def flight_header(flight_name):
    # flight-01a-ellipse → TII ELLIPSE 01a TRACK
    # flight-13p-trackRATM → TII TRACKRATM 13p TRACK
    parts = flight_name.split('-')          # ['flight', '01a', 'ellipse']
    num_id = parts[1]                       # '01a' or '13p'
    shape  = '-'.join(parts[2:]).upper()    # 'ELLIPSE', 'LEMNISCATE', 'TRACKRATM'
    return f"######### TII {shape} {num_id} TRACK #########"


def main():
    parser = argparse.ArgumentParser(
        description="Compute gate poses [x, y, z, yaw_rad] from flight marker data.")
    parser.add_argument('--flight', required=True, help="e.g. flight-01a-ellipse")
    args = parser.parse_args()

    flight_type = "piloted" if "p-" in args.flight else "autonomous"
    flight_dir  = os.path.join("..", "data", flight_type, args.flight)
    csv_dir     = os.path.join(flight_dir, "csv_raw")

    csv_files = glob(os.path.join(csv_dir, f"gate_corners_{args.flight}.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No gate_corners CSV found in {csv_dir}")
    df = pd.read_csv(csv_files[0])

    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in df.columns if col.startswith('gate') and '_marker1_x' in col
    })

    print(flight_header(args.flight))
    print("gates_poses:")
    for gate_id in gate_ids:
        corners = np.array([
            [df[f'gate{gate_id}_marker{m}_{ax}'].mean() for ax in ('x', 'y', 'z')]
            for m in range(1, 5)
        ])
        center = corners.mean(axis=0)

        _, _, Vt = np.linalg.svd(corners - center)
        normal   = Vt[2]
        n_xy     = np.array([normal[0], normal[1]])
        n_xy    /= np.linalg.norm(n_xy)
        yaw      = np.arctan2(n_xy[1], n_xy[0])

        print(f"  gate{gate_id - 1}: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}, {yaw:.5f}]")


if __name__ == '__main__':
    main()
