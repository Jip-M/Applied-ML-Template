import subprocess
import json
import os
import numpy as np
import pandas as pd
import requests
from pathlib import Path

metadata_dir = "dataset/metadata/grp_batsq_A"
BASE_DIR = Path(os.getcwd()).resolve()

def download_metadata() -> None:
    """
    Download metadata from Xeno-Canto API for bat species.
    """
    subprocess.run(["xeno-canto", "-m", "grp:\"bats\"", "q:\"A\""])
    with open(metadata_dir + "/page1.json", 'r') as file:
        api_data = json.load(file)
        print("API message:", api_data["message"])
        print("Number of recordings:", api_data["numRecordings"])
        print("Number of pages:", api_data["numPages"])

def load_metadata(species_selection: list[str]) -> pd.DataFrame:
    """
    Load metadata from JSON files in the specified directory and convert it to a DataFrame.
    Args:
        species_selection (list[str]): List of species names to filter the metadata DataFrame on.
    Returns:
        pd.DataFrame: DataFrame containing the metadata of bat recordings.
    """
    frames: list[pd.DataFrame] = []
    for entry in os.scandir(metadata_dir):
        if entry.is_file():
            with open(entry, 'r') as file:
                api_data = json.load(file)
                df = pd.DataFrame(api_data['recordings'])
                frames.append(df)

    all_data = pd.concat(frames)

    df = convert_columns(all_data)
    df = selection(df, species_selection)

    return df

def to_seconds(x: str) -> float:
    """
    Convert a string in the format "MM:SS" to seconds.
    Args:
        x (str): String representing time in "MM:SS" format.
    Returns:    
        float: Time in seconds.
    """
    mins, secs = map(float, x.split(':'))
    return mins * 60 + secs

def convert_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns of the DataFrame to more readable names and formats.
    Args:
        df (pd.DataFrame): DataFrame with metadata of bat recordings.
    Returns:
        pd.DataFrame: DataFrame with converted column names and formats.
    """
    df['length'] = df['length'].apply(to_seconds)
    df = df.rename(
        columns={"gen": "genus", "sp": "species", "en": "english_name", "cnt": "country", "type": "call_type",
                 "length": "audio_length", "dvc": "device", "mic": "microphone"})
    df = df[["id", "genus", "species", "english_name", "country", "call_type", "sex", "audio_length", "device",
                     "microphone"]]
    return df

def raw_url(file_id: int) -> str:
    return f"https://xeno-canto.org/{file_id}/download"

def raw_path(file_id: int) -> Path:
    return BASE_DIR / "data" / "raw" / f"{file_id}.wav"

def data_path(file_id: int, segment_id: int) -> Path:
    return BASE_DIR / "data" / "cleaned" / f"{file_id}_{segment_id}.csv"

def label_path() -> Path:
    return BASE_DIR / "data" / "labels.csv"

def create_folders() -> None:
    """
    Create necessary directories for storing data and trained models.
    """
    os.makedirs(BASE_DIR / "data" / "cleaned", exist_ok=True)
    os.makedirs(BASE_DIR / "data" / "raw", exist_ok=True)
    os.makedirs(BASE_DIR / "trained_model", exist_ok=True)

def download_file(file_id: int) -> None:
    """
    Download a file from Xeno-Canto using its ID and save it to the data directory.
    Args:
        file_id (int): ID of the file to download.
    """
    try:
        url = raw_url(file_id)
        file_path = raw_path(file_id)
        os.makedirs(file_path.parent, exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Failed to download {file_id}: {e}")

def selection(df: pd.DataFrame, selected_species: list[str]) -> pd.DataFrame:
    """
    Select samples from the DataFrame based on the provided species list and call type.
    Args:
        df (pd.DataFrame): DataFrame with metadata of bat recordings.
        selected_species (list[str]): List of species names to filter the DataFrame on.
    Returns:
        pd.DataFrame: Filtered DataFrame with samples of the selected species.
    """
    samples = df.loc[(df['species'].isin(selected_species)) & (df["call_type"] == "echolocation")].reset_index()

    stats = samples.groupby(["genus", "species", "english_name"])["species"].agg(["count"])
    stats.sort_values("count", ascending=False, inplace=True)
    stats.reset_index()
    np.set_printoptions(threshold=np.inf)
    return samples

def download_files(df: pd.DataFrame) -> None:
    """
    Download files from Xeno-Canto based on the provided DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with metadata of bat recordings.
    """
    for index, row in df.iterrows():
        file_id = row["id"]
        download_file(file_id)

