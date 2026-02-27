import os
import urllib.request
import gzip
import shutil

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

# Map of (hyphen_name, dot_name, compressed_name)
FILES = [
    ("train-images-idx3-ubyte", "train-images.idx3-ubyte", "train-images-idx3-ubyte.gz"),
    ("train-labels-idx1-ubyte", "train-labels.idx1-ubyte", "train-labels-idx1-ubyte.gz"),
    ("t10k-images-idx3-ubyte",  "t10k-images.idx3-ubyte",  "t10k-images-idx3-ubyte.gz"),
    ("t10k-labels-idx1-ubyte",  "t10k-labels.idx1-ubyte",  "t10k-labels-idx1-ubyte.gz")
]

def main():
    # Find the project root and data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    # Create data directory if not exists
    if not os.path.exists(data_dir):
        print(f"Creating directory: {data_dir}")
        os.makedirs(data_dir)

    all_exist = True

    for hyphen_name, dot_name, gz_name in FILES:
        hyphen_path = os.path.join(data_dir, hyphen_name)
        dot_path = os.path.join(data_dir, dot_name)
        gz_path = os.path.join(data_dir, gz_name)

        # Check if either the hyphenated OR dotted version exists
        if os.path.exists(hyphen_path):
            print(f"[OK] Found {hyphen_name}")
            continue
        if os.path.exists(dot_path):
            print(f"[OK] Found {dot_name}")
            continue
        
        all_exist = False
        url = BASE_URL + gz_name
        print(f"[DOWNLOADING] {gz_name}...")
        
        try:
            urllib.request.urlretrieve(url, gz_path)
            
            print(f"[EXTRACTING]  {gz_name} -> {hyphen_name}")
            with gzip.open(gz_path, 'rb') as f_in:
                # We default to saving it with the official hyphenated name
                with open(hyphen_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to download or extract {gz_name}: {e}")
            return

    if all_exist:
        print("\nAll MNIST data files are already present and ready to go!")
    else:
        print("\nDataset successfully downloaded and extracted into the /data folder.")

if __name__ == "__main__":
    main()