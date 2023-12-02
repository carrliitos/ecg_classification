import os

def rename_and_move_files(root_path):
    for person_folder in os.listdir(root_path):
        person_folder_path = os.path.join(root_path, person_folder)
        
        if os.path.isdir(person_folder_path):
            for csv_file in os.listdir(person_folder_path):
                if csv_file.endswith(".csv"):
                    old_path = os.path.join(person_folder_path, csv_file)
                    new_name = f"{person_folder}_{csv_file}"
                    new_path = os.path.join(root_path, new_name)
                    
                    os.rename(old_path, new_path)

            os.rmdir(person_folder_path)

if __name__ == "__main__":
    processed_folder_path = "./data/processed/ecg-id-database-1.0.0"
    rename_and_move_files(processed_folder_path)
