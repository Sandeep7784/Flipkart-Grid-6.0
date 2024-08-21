import csv
import random
from vectorDB.measurement_index import pinecone_vector_store

def read_csv_to_dict(filename):
    data_dict = {}
    
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        
        for row in csv_reader:
            user_id = random.randint(1000, 9999)
            data_dict[user_id] = {headers[i]: row[i] for i in range(len(headers))}

    return data_dict

filename = 'data.csv'
data = read_csv_to_dict(filename)
pinecone_vector_store(data)