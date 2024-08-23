import csv
import random

def read_csv_to_dict(filename):
    data_dict = {}
    company_dict = {}
    
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)

        headers.insert(1, 'company_id')

        for row in csv_reader:
            company_name = row[1]

            if company_name not in company_dict:
                company_dict[company_name] = random.randint(1, 1000)

            company_id = company_dict[company_name]

            row.insert(1, str(company_id))

            user_id = row[0]
            data_dict[user_id] = {headers[i]: row[i] for i in range(len(headers))}
    
    # Write the updated data back to the CSV file
    with open(filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(headers)
        for user_id, user_data in data_dict.items():
            csv_writer.writerow([user_data[header] for header in headers])

    return data_dict

filename = 'data.csv'
data = read_csv_to_dict(filename)