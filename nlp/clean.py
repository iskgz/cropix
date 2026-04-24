input_file = "big_dataset.csv"
output_file = "clean_dataset.csv"

seen = set()

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    for line in f_in:
        line_clean = line.strip()
        
        if line_clean not in seen:
            f_out.write(line)
            seen.add(line_clean)