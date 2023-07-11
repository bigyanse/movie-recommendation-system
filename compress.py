import gzip

def compress_pickle(input_filename, output_filename):
    with open(input_filename, 'rb') as file:
        data = file.read()

    with gzip.open(output_filename, 'wb') as gz_file:
        gz_file.write(data)

    print(f"The pickle file '{input_filename}' has been compressed to '{output_filename}'.")

def decompress_pickle(input_filename, output_filename):
    with gzip.open(input_filename, 'rb') as gz_file:
        data = gz_file.read()

    with open(output_filename, 'wb') as file:
        file.write(data)

    print(f"The compressed pickle file '{input_filename}' has been decompressed to '{output_filename}'.")

compress_pickle("similarity.pkl", "similarity.pkl.gz")
