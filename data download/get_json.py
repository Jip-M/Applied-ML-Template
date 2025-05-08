import json
import glob


def find_all_file_keys(obj, target_key: str):
    """
    This function gets all the occurences of the word "file" inside the
    metadata. This way, we get all the links that let us access the
    recordings by downloading them in a .wav format.
    """
    results = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                results.append(value)

            results.extend(find_all_file_keys(value, target_key))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_all_file_keys(item, target_key))
    return results


all_the_links = []
all_names = []
folder_path = 'Applied-ML-Template/notebooks/dataset/metadata/grp_batsq_A'

# since we have a folder of metadata, start from the folder
for filename in glob.glob(folder_path + "/*.json"):
    # and then, we select each json file there is avaiable inside the folder
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Process your metadata here by adding all the links from all the json pages
        file_links = find_all_file_keys(data, target_key="file")
        
        file_names = find_all_file_keys(data, target_key="id")

    # since there are 3 pages in our case, we have to do this step 3 times, so we append
    # the lists in a bigger list
    all_the_links.append(file_links)
    all_names.append(file_names)


with open("Applied-ML-Template/data download/wav_links.txt", "wb") as f:
    for links in all_the_links:
        for link in links:
            # here we just write all the links inside a file to be able to access each one of them
            # and download them individually
            f.write((link + "\n").encode('utf-8'))
            

with open("Applied-ML-Template/data download/common_names.txt", "wb") as f:
    for names in all_names:
        for name in names:
            # here we just write all the names inside a file
            f.write((name + "\n").encode('utf-8'))





