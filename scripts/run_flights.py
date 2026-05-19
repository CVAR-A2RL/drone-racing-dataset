#!/usr/bin/env python3

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).parent.resolve()
DUMMY_TEMPLATE = SCRIPTS_DIR / 'rosbags_output' / 'dummy_bag' / 'imu_cam_bag'
OUTPUT_DIR = SCRIPTS_DIR / 'rosbags_output'

BOOL_FLAGS = [
    'compressed', 'as2', 'rectified', 'rectified-points', 'fix-rotation',
    'static-map', 'static-gates', 'computed-corners', 'labels-from-3d',
]
SCALAR_KEYS = ['interior-size', 'exterior-size', 'gate-depth', 'arm', 'offboard']
LIST_KEYS = ['noise-std', 'detections-downsample', 'pose-noise-std', 'pose-downsample']


def build_cmd(flight_name, global_cfg, flight_cfg):
    cmd = [sys.executable, 'create_std_bag.py', '--flight', flight_name]

    for key in BOOL_FLAGS:
        if global_cfg.get(key):
            cmd.append(f'--{key}')

    for key in SCALAR_KEYS:
        if key in global_cfg:
            cmd += [f'--{key}', str(global_cfg[key])]

    for key in LIST_KEYS:
        if key in global_cfg:
            cmd += [f'--{key}'] + [str(v) for v in global_cfg[key]]

    if 'start' in flight_cfg:
        cmd += ['--start', str(flight_cfg['start'])]
    if 'end' in flight_cfg:
        cmd += ['--end', str(flight_cfg['end'])]

    return cmd


def run_bag_gen(flight_name, cmd):
    flight_type = 'piloted' if 'p-' in flight_name else 'autonomous'
    bag_out = SCRIPTS_DIR.parent / 'data' / flight_type / flight_name / 'imu_cam_bag'
    if bag_out.exists():
        shutil.rmtree(bag_out)

    print(f'  Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR)
    if result.returncode != 0:
        print(f'  WARNING: create_std_bag.py exited with code {result.returncode} for {flight_name}')
        return False
    return True


def copy_structure(flight_name):
    dst = OUTPUT_DIR / flight_name / 'imu_cam_bag'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(DUMMY_TEMPLATE, dst)

    # Replace inner dummy rosbag with the generated one
    flight_type = 'piloted' if 'p-' in flight_name else 'autonomous'
    generated_bag = SCRIPTS_DIR.parent / 'data' / flight_type / flight_name / 'imu_cam_bag'
    inner = dst / 'imu_cam_bag'
    shutil.rmtree(inner)
    shutil.copytree(generated_bag, inner)

    return dst / 'config'


def patch_gates_config(config_dir, gates_poses):
    lines = ['gates_poses:\n']
    for name, vals in gates_poses.items():
        lines.append(f'  {name}: {vals}\n')
    (config_dir / 'gates_config.yaml').write_text(''.join(lines))


def patch_state_estimator(config_dir, position):
    path = config_dir / 'state_estimator_config.yaml'
    lines = path.read_text().splitlines(keepends=True)

    in_block = False
    xyz_replaced = {'x': False, 'y': False, 'z': False}
    done = False
    out = []

    for line in lines:
        if '#### STARTING POSITION ####' in line:
            in_block = True

        if in_block and not done:
            if not xyz_replaced['x'] and re.match(r'\s+x:\s', line):
                line = re.sub(r'(x:)\s+\S+', f'x: {position["x"]}', line)
                xyz_replaced['x'] = True
            elif not xyz_replaced['y'] and re.match(r'\s+y:\s', line):
                line = re.sub(r'(y:)\s+\S+', f'y: {position["y"]}', line)
                xyz_replaced['y'] = True
            elif not xyz_replaced['z'] and re.match(r'\s+z:\s', line):
                line = re.sub(r'(z:)\s+\S+', f'z: {position["z"]}', line)
                xyz_replaced['z'] = True
                done = True

        out.append(line)

    path.write_text(''.join(out))


def detection_topic_variants(noise_stds, downsample_rates):
    """Yield (topic_suffix, filename_suffix) for all detection topic variants."""
    def fmt(v):
        return str(v).replace('.', '_')

    yield '', ''  # base: debug/detected_gates_data → detection_config.yaml

    for ds in downsample_rates:
        yield f'_ds_{fmt(ds)}', f'_ds_{fmt(ds)}'

    for std in noise_stds:
        yield f'_{fmt(std)}', f'_{fmt(std)}'
        for ds in downsample_rates:
            yield f'_{fmt(std)}_ds_{fmt(ds)}', f'_{fmt(std)}_ds_{fmt(ds)}'


def generate_detection_configs(config_dir, noise_stds, downsample_rates):
    base_path = config_dir / 'detection_config.yaml'
    base_text = base_path.read_text()
    base_topic = 'debug/detected_gates_data'

    for topic_suffix, file_suffix in detection_topic_variants(noise_stds, downsample_rates):
        topic = base_topic + topic_suffix
        text = base_text.replace(
            f'"{base_topic}"',
            f'"{topic}"',
        )
        filename = f'detection_config{file_suffix}.yaml'
        (config_dir / filename).write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description='Run create_std_bag.py for all flights in metadata and package outputs.')
    parser.add_argument('--metadata', default='flight_metadata.yaml',
                        help='Path to flight_metadata.yaml (default: flight_metadata.yaml)')
    parser.add_argument('--skip-bag-gen', action='store_true',
                        help='Skip create_std_bag.py; only copy structure and patch configs')
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = SCRIPTS_DIR / metadata_path

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    global_cfg = metadata.get('global', {})
    flights = metadata.get('flights', {})

    for i, (flight_name, flight_cfg) in enumerate(flights.items(), 1):
        print(f'\n[{i}/{len(flights)}] {flight_name}')

        if not args.skip_bag_gen:
            cmd = build_cmd(flight_name, global_cfg, flight_cfg)
            ok = run_bag_gen(flight_name, cmd)
            if not ok:
                print(f'  Skipping packaging for {flight_name} due to bag generation failure.')
                continue

        print('  Copying structure...')
        config_dir = copy_structure(flight_name)

        print('  Patching gates_config.yaml...')
        patch_gates_config(config_dir, flight_cfg['gates_poses'])

        print('  Patching state_estimator_config.yaml...')
        patch_state_estimator(config_dir, flight_cfg['position'])

        print('  Generating detection_config variants...')
        generate_detection_configs(
            config_dir,
            global_cfg.get('noise-std', []),
            global_cfg.get('detections-downsample', []),
        )

        print(f'  Done → rosbags_output/{flight_name}/imu_cam_bag/')

    print('\nAll flights processed.')


if __name__ == '__main__':
    main()
