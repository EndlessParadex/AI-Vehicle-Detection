import os

LABEL_ROOT = "dataset_plate_1class"

def convert_to_single_class_safe():
    converted = 0
    skipped_empty = 0
    skipped_invalid = 0

    for root, _, files in os.walk(LABEL_ROOT):
        if os.path.basename(root) != "labels":
            continue

        for file in files:
            if not file.endswith(".txt"):
                continue

            path = os.path.join(root, file)

            with open(path, "r") as f:
                lines = f.readlines()

            if len(lines) == 0:
                skipped_empty += 1
                continue  # BIARKAN KOSONG

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    skipped_invalid += 1
                    continue

                new_lines.append("0 " + " ".join(parts[1:5]) + "\n")

            if len(new_lines) == 0:
                skipped_invalid += 1
                continue  # JANGAN TIMPA

            with open(path, "w") as f:
                f.writelines(new_lines)

            converted += 1

    print("✅ Conversion done")
    print(f"Converted files : {converted}")
    print(f"Skipped empty   : {skipped_empty}")
    print(f"Skipped invalid : {skipped_invalid}")

if __name__ == "__main__":
    convert_to_single_class_safe()
