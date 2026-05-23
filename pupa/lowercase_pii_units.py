import pandas as pd
from pathlib import Path

def main():
    pupa_dir = Path("/home/min/PAPILLON/pupa")
    files = ["PUPA_New_repii.csv", "PUPA_TNB_repii.csv"]

    for file_name in files:
        file_path = pupa_dir / file_name
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
        
        print(f"Processing {file_name}...")
        df = pd.read_csv(file_path)
        
        def to_lowercase(val):
            if not isinstance(val, str):
                return val
            # 소문자로 변환하되 양끝 공백이 있을 경우 정리
            return "||".join(item.strip().lower() for item in val.split("||"))
            
        df["pii_units"] = df["pii_units"].apply(to_lowercase)
        
        df.to_csv(file_path, index=False)
        print(f"Successfully converted 'pii_units' in {file_name} to lowercase.")

if __name__ == "__main__":
    main()
