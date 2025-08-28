import os
import shutil
import json

def recursive_replace(item, old_base, new_base):
    if isinstance(item, str):
        return item.replace(old_base, new_base)
    elif isinstance(item, list):
        return [recursive_replace(elem, old_base, new_base) for elem in item]
    elif isinstance(item, dict):
        return {k: recursive_replace(v, old_base, new_base) for k, v in item.items()}
    else:
        return item

def copy_and_update_jsons(
    source_dir,
    suffix="_updated",
    old_base="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/",
    new_base="/media/jaume/T7/"
):
    parent_dir = os.path.dirname(source_dir)
    folder_name = os.path.basename(source_dir)
    new_folder_name = folder_name + suffix
    new_dir = os.path.join(parent_dir, new_folder_name)

    # Step 1: Copy the whole folder
    shutil.copytree(source_dir, new_dir)
    print(f"Copied folder to: {new_dir}")

    # Step 2: Update all JSON files in the new folder
    for filename in os.listdir(new_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(new_dir, filename)

            with open(json_path, 'r') as f:
                data = json.load(f)

            updated_data = recursive_replace(data, old_base, new_base)

            with open(json_path, 'w') as f:
                json.dump(updated_data, f, indent=2)

            print(f"Updated: {filename}")

# Example usage:
copy_and_update_jsons(
    source_dir="/media/jaume/T7/data/splits_final/task3/dino_experiments/fomo-task3-2ch-mimic",
    suffix="_local"
)
