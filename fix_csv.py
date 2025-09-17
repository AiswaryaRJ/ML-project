import csv

input_file = "generated_dataset.csv"
output_file = "generated_dataset_fixed.csv"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    
    # Use csv.reader with default quoting
    reader = csv.reader(infile, delimiter=',', quotechar='"')
    writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for line_number, row in enumerate(reader, start=1):
        if len(row) == 2:
            writer.writerow(row)
        elif len(row) > 2:
            # Merge all extra fields into the second column
            fixed_row = [row[0], ",".join(row[1:])]
            writer.writerow(fixed_row)
            print(f"Fixed line {line_number}: {row} -> {fixed_row}")
        else:
            # Skip lines that have fewer than 2 fields
            print(f"Skipped line {line_number}: {row}")

print(f"CSV fixed successfully. Saved as {output_file}")
