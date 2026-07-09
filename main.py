"""Webcam ile yuz ifadesinden duygu tahmini - giris noktasi."""

import argparse
from pathlib import Path

from src.app import App


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam ile yuz ifadesinden duygu tahmini")
    parser.add_argument("--camera", type=int, default=0, help="Kamera indeksi (varsayilan: 0)")
    parser.add_argument(
        "--calibration-file",
        type=str,
        default="calibration_profiles/default.json",
        help="Kalibrasyon profili dosya yolu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = App(camera_index=args.camera, calibration_path=Path(args.calibration_file))
    app.run()


if __name__ == "__main__":
    main()
