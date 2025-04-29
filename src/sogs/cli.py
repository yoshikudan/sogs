import os
from dataclasses import dataclass
from pathlib import Path
import tyro
from .sogs_compression import read_ply, run_compression

@dataclass
class Config:
    ply: Path
    output_dir: Path

def main():
	cfg = tyro.cli(Config)
	os.makedirs(cfg.output_dir, exist_ok=True)
	splats = read_ply(cfg.ply)
	run_compression(cfg.output_dir, splats)

if __name__ == "__main__":
	main()