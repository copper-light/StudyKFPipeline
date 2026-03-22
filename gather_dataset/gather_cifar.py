import os
from tqdm import tqdm
from glob import glob
from postgres import Postgres
from datetime import datetime, timedelta


conn_str = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:1234@localhost:5432/mlops"
)

# def create_table(db):
#     """Creates the images table if it doesn't exist."""
#     db.run(
#         "CREATE TABLE IF NOT EXISTS images (id serial PRIMARY KEY, filename text, label text, data bytea)"
#     )


def save_to_db():
    # Database connection string from environment variable or default
    # For local development, you might use: "postgresql://user:password@localhost:5432/dbname"

    try:
        db = Postgres(conn_str)

        image_paths = glob("datasets/cifar10/train/**/**.png")

        if not image_paths:
            print("No images found in dataset/cifar10/")
            return

        for p in tqdm(image_paths):
            filename = os.path.basename(p)
            # Assuming label is the directory name
            label = os.path.dirname(p).split("/")[-1]

            with open(p, "rb") as f:
                image_data = f.read()

            print(f"Saving {filename} ({label}) to DB...")
            db.run(
                "INSERT INTO cifar10_files (path, data) VALUES (%s, %s)",
                (os.path.join(label, filename), image_data),
            )

        print("Success: All images saved to database.")

    except Exception as e:
        print(f"Error connecting to database or saving images: {e}")


def labeling():
    try:
        db = Postgres(conn_str)

        # Fetch data from source table
        print("Fetching data from cifar10_files...")

        # Ensure is_processed column exists
        db.run(
            "ALTER TABLE cifar10 ADD COLUMN IF NOT EXISTS is_processed BOOLEAN DEFAULT FALSE"
        )

        rows = db.all("SELECT path, data FROM cifar10_files")

        if not rows:
            print("No data found in cifar10_files.")
            return

        cnt = 0
        updated_date = datetime.now()
        for row in tqdm(rows):
            path = row.path
            image_data = row.data

            # Preprocessing: Extract label from path (e.g., 'airplane/123.png')
            label = path.split("/")[0] if "/" in path else "unknown"

            print(f"Processing and saving {path} with label {label}...")

            if cnt > 0 and cnt % 10000 == 0:
                updated_date += timedelta(days=1)
            cnt += 1

            # Save to target table
            db.run(
                "INSERT INTO cifar10 (path, label, updated, is_processed) VALUES (%s, %s, %s, %s)",
                (path, label, updated_date, False),
            )

        print("Success: Data preprocessed and stored in cifar10.")

    except Exception as e:
        print(f"Error during labeling and storage: {e}")


if __name__ == "__main__":
    save_to_db()
    labeling()
