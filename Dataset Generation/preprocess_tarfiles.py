import os
import tarfile
import gzip
import shutil
from tqdm import tqdm


def standardize_tex_file(extract_path, dir_name):
    try:
        files_in_dir = os.listdir(extract_path)
    except FileNotFoundError:
        return

    if not files_in_dir:
        return  # Empty extraction

    # Archive contained exactly one file. It must be the source.
    if len(files_in_dir) == 1:
        original_filename = files_in_dir[0]
        old_path = os.path.join(extract_path, original_filename)
        new_path = os.path.join(extract_path, dir_name + '.tex')
        if old_path != new_path:
            os.rename(old_path, new_path)
        return



    # Check if a perfectly named .tex file already exists.
    if f"{dir_name}.tex" in files_in_dir:
        return

    tex_files = [f for f in files_in_dir if f.lower().endswith('.tex')]

    #There is exactly one .tex file, making it the obvious main source.
    if len(tex_files) == 1:
        original_tex_filename = tex_files[0]
        old_path = os.path.join(extract_path, original_tex_filename)
        new_path = os.path.join(extract_path, dir_name + '.tex')
        # Just in case the filename is the same but with different casing
        if os.path.normcase(old_path) != os.path.normcase(new_path):
            os.rename(old_path, new_path)
        return




def final_robust_extractor_V5(source_dir, output_dir):
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_files = os.listdir(source_dir)
    archives = [f for f in all_files if f.endswith(('.tar.gz', '.gz'))]

    if not archives:
        print(f"No archives found in '{source_dir}'.")
        return

    error_log_path = os.path.join(output_dir, 'extraction_errors_V5.txt')
    failed_files = []

    print(f"Found {len(archives)} archives. Starting V5 robust extraction...")

    for file_name in tqdm(archives, desc="Extracting archives", unit="file"):
        file_path = os.path.join(source_dir, file_name)

        if file_name.endswith('.tar.gz'):
            dir_name = file_name[:-7]
        else:  # .gz
            dir_name = file_name[:-3]

        extract_path = os.path.join(output_dir, dir_name)

        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        os.makedirs(extract_path, exist_ok=True)

        extracted_successfully = False

        # Attempt 1: Standard .tar.gz
        try:
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)

            if os.listdir(extract_path):
                extracted_successfully = True
                standardize_tex_file(extract_path, dir_name)

        except (tarfile.ReadError, EOFError, tarfile.TarError, IOError):
            pass

            # Attempt 2: Single gzipped file
        if not extracted_successfully:
            try:
                with gzip.open(file_path, 'rb') as f_in:
                    content = f_in.read()

                if content:
                    output_file_path = os.path.join(extract_path, dir_name + '.tex')
                    with open(output_file_path, 'wb') as f_out:
                        f_out.write(content)
                    extracted_successfully = True
            except (gzip.BadGzipFile, OSError, EOFError):
                pass

                # Attempt 3: Plain .tar file
        if not extracted_successfully:
            try:
                with tarfile.open(file_path, 'r:') as tar:
                    tar.extractall(path=extract_path)

                if os.listdir(extract_path):
                    extracted_successfully = True
                    standardize_tex_file(extract_path, dir_name)

            except (tarfile.ReadError, tarfile.TarError, IOError):
                pass

        if not extracted_successfully:
            failed_files.append(f"{file_name}: Failed all extraction methods.")
            if os.path.exists(extract_path) and not os.listdir(extract_path):
                os.rmdir(extract_path)

    print("\nExtraction process completed.")
    if failed_files:
        print(f"There were {len(failed_files)} errors after all attempts.")
        print(f"A detailed log has been saved to: '{error_log_path}'")
        with open(error_log_path, 'w') as f:
            f.write("The following files failed all extraction attempts:\n" + "\n".join(failed_files))
    else:
        print("All archives were extracted successfully!")


if __name__ == '__main__':
    SOURCE_DIRECTORY = "/home/harris/PycharmProjects/PythonProject/math_src_1600-1900"
    OUTPUT_DIRECTORY = "/home/harris/PycharmProjects/PythonProject/math_src_1600-1900_EXTRACTED"

    print(f"Source: {SOURCE_DIRECTORY}")
    print(f"Output: {OUTPUT_DIRECTORY}")


    final_robust_extractor_V5(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)