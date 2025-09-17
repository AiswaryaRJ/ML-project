import csv

input_file = "generated_dataset.csv"
output_file = "generated_dataset_safe.csv"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile, delimiter=',', quotechar='"')
    writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for line_number, row in enumerate(reader, start=1):
        if len(row) == 2:
            writer.writerow(row)
        elif len(row) > 2:
            # Merge extra columns into the second field
            fixed_row = [row[0], ",".join(row[1:])]
            writer.writerow(fixed_row)
            print(f"Fixed line {line_number}: {row} -> {fixed_row}")
        elif len(row) == 1:
            # Line has only 1 field, add empty second field
            fixed_row = [row[0], ""]
            writer.writerow(fixed_row)
            print(f"Added empty column at line {line_number}: {row} -> {fixed_row}")
        else:
            # Completely empty line
            writer.writerow(["", ""])
            print(f"Empty line at {line_number} preserved")

print(f"All lines processed. Safe CSV saved as {output_file}")
