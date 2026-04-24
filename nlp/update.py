fixed_lines = []

with open("big_dataset.csv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")

        if len(parts) > 2:
            text = ",".join(parts[:-1])
            label = parts[-1]
            fixed_lines.append(f'"{text}",{label}')
        else:
            fixed_lines.append(line.strip())

with open("big_dataset_fixed.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(fixed_lines))

print("🔥 Düzeltildi: big_dataset_fixed.csv")