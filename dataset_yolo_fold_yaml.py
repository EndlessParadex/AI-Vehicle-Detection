import yaml
import os

# Nama yaml asli (hanya untuk ambil daftar kelas)
SOURCE_YAML = "data_vehicle.yaml"

# Lokasi dataset 5-fold
DATASET_BASE = "dataset_5fold/dataset_5fold"

def main():
    # Baca YAML asli untuk ambil nc dan names
    with open(SOURCE_YAML, "r") as f:
        base = yaml.safe_load(f)

    names = base.get("names", [])
    nc = base.get("nc", len(names))

    print(f"Loaded original YAML:")
    print(f" - nc    = {nc}")
    print(f" - names = {names}")

    # Generate 5 YAML
    for i in range(1, 6):
        yaml_data = {
            "train": f"{DATASET_BASE}/fold{i}/train/images",
            "val": f"{DATASET_BASE}/fold{i}/val/images",
            "test": None,  # YOLO menerima null
            "nc": nc,
            "names": names
        }

        out_file = f"data_fold{i}.yaml"
        with open(out_file, "w") as f:
            yaml.dump(yaml_data, f, sort_keys=False)

        print(f"Generated: {out_file}")

    print("\n✔ Semua YAML 5-fold berhasil dibuat dengan format yang kamu minta.\n")


if __name__ == "__main__":
    main()
