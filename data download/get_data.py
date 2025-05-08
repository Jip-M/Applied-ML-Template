import os
import requests

# Directory to save the downloaded files
save_dir = "Applied-ML-Template/data download"
os.makedirs(os.path.join(save_dir, "wavs"), exist_ok=True)


check = []
with open('Applied-ML-Template/data download/common_names.txt', 'r') as file:
    for line in file:
        line = line.rstrip('\n')
        if line not in check:
            check.append(line)

for elem in check:
    os.makedirs(os.path.join("Applied-ML-Template/data download/wavs", elem), exist_ok=True)

# count = 0
with open('Applied-ML-Template/data download/wav_links.txt', 'r') as f1, open('Applied-ML-Template/data download/common_names.txt', 'r') as f2:
    for line1, line2 in zip(f1, f2):
        # Remove trailing newlines to avoid double-spacing
        line1 = line1.rstrip('\n')
        line2 = line2.rstrip('\n')
        filename = "Applied-ML-Template/data download/wavs/" + line2

        print(f"Downloading {line1} ...")
        try:
            response = requests.get(line1, stream=True)
            response.raise_for_status()

            # filename = filename + f"{count}" + ".wav"
            # 
            print("TESTSTSTSTSTS", filename)
            filename = os.path.join(filename, line2 + ".wav")
            # count += 1
            with open(filename, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        out_file.write(chunk)
            print(f"Saved to {filename}")
        except Exception as e:
            print(f"Failed to download {line1}: {e}")


