import csv
import random
# from vectorDB.measurement_index import pinecone_vector_store

def read_csv_to_dict(filename):
    data_dict = {}
    
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        
        # Add 'user_id' to the headers
        headers.insert(0, 'user_id')
        
        for row in csv_reader:
            user_id = random.randint(1000, 9999)
            # Add user_id to the row data
            row.insert(0, str(user_id))
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
# pinecone_vector_store(data)