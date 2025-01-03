import os
import tarfile
from tarfile import TarFile
import numpy as np
from typing import BinaryIO
import dask.array as da
import io

def binary_to_signal(bin_file: BinaryIO,
                    num_channels: int = 385,
                    chunk_size: int = 100000,
                    dtype: np.dtype = np.int16) -> np.ndarray:
    """Convert binary file data to a numpy array of signal data.

    This function reads binary data in chunks and converts it to a numpy array
    of signal data. It handles large files efficiently by processing them in
    chunks rather than loading the entire file into memory at once.

    Args:
        bin_file: A binary file object containing the signal data.
            Must be opened in binary mode ('rb') and support read operations.
        num_channels: Number of channels in the data.
            Each sample in the binary file contains data for this many channels.
            Defaults to 385.
        chunk_size: Number of samples to read at a time.
            Used to control memory usage when reading large files.
            Defaults to 100000.
        dtype: Data type of the binary data.
            Should match the format of the binary file.
            Defaults to np.int16.

    Returns:
        A 2D numpy array containing the signal data.
        Shape is (num_channels, num_samples) where num_samples depends on
        the length of the input file.

    Raises:
        Exception: If there is an error processing the binary data.
            The original error message is included in the exception.
    """
    try:
        all_data = []
        while True:
            if hasattr(bin_file, 'read'):
                data_chunk = np.frombuffer(
                    bin_file.read(chunk_size * num_channels * np.dtype(dtype).itemsize),
                    dtype=dtype
                )
                if data_chunk.size == 0:
                    break
                data_chunk = data_chunk.reshape(-1, num_channels).T
                all_data.append(data_chunk)
            else:
                print('Binary data not read')

        reshaped_data = np.hstack(all_data)
        return reshaped_data

    except Exception as e:
        raise Exception(f"Error processing binary data: {str(e)}")

def npy_loader(tar:TarFile,filename:str)-> np.ndarray:
    '''
    Numpy loader function for .npy in tarball (.tar) packages.

    :param filename: str
    :return: np.ndarray
    '''
    try:
        npy_file = tar.extractfile(filename)
        if npy_file is not None:
            npy_file_content = npy_file.read()

            # Check file size to confirm it's not empty or corrupted
            if len(npy_file_content) == 0:
                raise ValueError(f"The .npy file '{filename}' is empty or corrupted.")

            # Load .npy file from memory using BytesIO
            np_data = np.load(io.BytesIO(npy_file_content))
            return np_data
        else:
            raise FileNotFoundError(f"Could not find or extract the file: {filename}")
    except Exception as e:
        print(f"Error reading .npy file: {e}")


def get_probe_signals(tar_path: str,
                     probe_select: int,
                     num_channels: int = 385,
                     chunk_size: int = 100000,
                     dtype: np.dtype = np.int16) -> np.ndarray:
    """Extract and process probe signals from a tar file.

    This function opens a tar file containing probe signal data, extracts
    the specified probe's binary data, and converts it to a numpy array.

    Args:
        tar_path: Path to the tar file containing probe signal data.
            Must be a valid path to an existing tar file.
        probe_select: Index of the probe to extract data from.
            Must be a valid index for the available probes in the tar file.
        num_channels: Number of channels in the probe data.
            Each sample contains data for this many channels.
            Defaults to 385. Last channel is synchronization data.
        chunk_size: Number of samples to process at a time.
            Used to control memory usage when reading large files.
            Defaults to 100000.
        dtype: Data type of the binary data.
            Should match the format of the stored data.
            Defaults to np.int16.

    Returns:
        A 2D numpy array containing the signal data for the selected probe.
        Shape is (num_channels, num_samples) where num_samples depends on
        the length of the probe data.

    Raises:
        IndexError: If probe_select is out of range for the available probes.
        tarfile.TarError: If there are issues reading the tar file.
        Exception: If there is an error processing the binary data.
    """

    # Verify the file exists
    try:
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found at: {tar_path}")
        elif not tar_path.endswith(".tar"):
            raise FileNotFoundError(f"{tar_path} is not a valid tarball file.")
        elif not tar_path.endswith("_lfp.tar"):
            raise FileNotFoundError(f"{tar_path} is not a valid LFP tarball file.")

        print('Reading probe signal data from tar file...')

        with tarfile.open(tar_path, 'r') as tar:
            bin_file_names = np.array(tar.getnames())
            print(bin_file_names)

            if probe_select is not None:
                if isinstance(probe_select, (int, np.integer)):
                    if probe_select >= len(bin_file_names)-1:
                        raise IndexError(
                            f"probe_select index {probe_select} is out of range "
                            f"for {len(bin_file_names)} probes"
                        )
                    bin_file_name = bin_file_names[probe_select]
            else:
                raise TypeError('probe_select must be an integer')

            bin_io = tar.extractfile(bin_file_name)
            reshaped_data = binary_to_signal(bin_io, num_channels, chunk_size, dtype)

    except tarfile.TarError as e:
        raise ValueError(f"Error reading tar file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")

    return da.from_array(reshaped_data, chunks=(16,-1))