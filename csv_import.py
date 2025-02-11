import csv

def load_csv_data(file_path):
    rpm = []
    flow = []
    pressure = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        
        for row in reader:
            if len(row) < 3:
                continue  # Skip incomplete rows
            try:
                rpm.append(float(row[0]))
                flow.append(float(row[1]))
                pressure.append(float(row[2]))
            except ValueError:
                continue  # Skip invalid rows

    return rpm, flow, pressure

# Example usage
file_path = "turbo_data.csv"
rpm, flow, pressure = load_csv_data(file_path)

print("RPM:", rpm)
print("Flow:", flow)
print("Pressure:", pressure)
