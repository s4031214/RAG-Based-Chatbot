import csv
import yaml  # pip install pyyaml

def csv_to_yaml(csv_path, yaml_path="output.yml"):
    urls_list = []

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Build dictionary only with the required fields
            entry = {
                "url": row.get("source_url", "").strip(),
                "title": row.get("title", "").strip(),
                "role": row.get("role", "").strip(),
                "category": row.get("category", "").strip(),
                "jurisdiction": row.get("jurisdiction", "").strip(),
            }
            urls_list.append(entry)

    # Create the YAML structure
    data = {"urls": urls_list}

    with open(yaml_path, "w", encoding="utf-8") as yamlfile:
        yaml.dump(data, yamlfile, allow_unicode=True, sort_keys=False)

    print(f"✅ YAML saved to {yaml_path}")


if __name__ == "__main__":
    csv_to_yaml("data/clean/md_metadata.csv", "urls.yml")
