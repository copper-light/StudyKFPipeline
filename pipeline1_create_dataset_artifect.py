import os
import tarfile
import io
import json
import datetime
from minio import Minio
from postgres import Postgres

# Database connection string
conn_str = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:1234@localhost:5432/mlops"
)

client = Minio(
    endpoint="localhost:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False,
)


def create_dataset(
    target_date_str: str, dataset_name: str = "cifar10", version: str = "v1"
):
    """
    Gathers unprocessed data from DB up to target_date, uploads to Minio as tar.gz,
    and updates labels_{version}.json.
    """
    db = Postgres(conn_str)

    target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d")
    q_target_date = target_date + datetime.timedelta(days=1)

    # 1. Fetch unprocessed data
    print(f"Fetching unprocessed data up to {target_date_str}...")
    query = """
        SELECT c.path, c.label, f.data, to_char(c.updated, 'YYYY-MM-DD') as updated
        FROM cifar10 c
        JOIN cifar10_files f ON c.path = f.path
        WHERE c.updated < %s AND c.is_processed = FALSE
    """
    rows = db.all(query, (q_target_date,))

    if not rows:
        print("No new data to process.")
        return

    # 2. Create tar.gz in memory
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for row in rows:
            file_data = row.data
            tarinfo = tarfile.TarInfo(name=row.path)
            tarinfo.size = len(file_data)
            tar.addfile(tarinfo, io.BytesIO(file_data))

    tar_data = tar_buffer.getvalue()
    tar_size = len(tar_data)
    tar_buffer.seek(0)

    # 3. Upload to Minio: {dataset_name}/{date}.gz.tar
    object_name = f"{dataset_name}/images/{target_date_str}.gz.tar"
    print(f"Uploading {object_name} to 'datasets' bucket...")

    if not client.bucket_exists("datasets"):
        client.make_bucket("datasets")

    client.put_object(
        "datasets",
        object_name,
        data=tar_buffer,
        length=tar_size,
        content_type="application/gzip",
    )

    # 4. Manage labels_{version}.json
    label_json_path = f"{dataset_name}/labels/labels_{version}.json"
    existing_labels = {"version": version, "image_tars": [], "labels": []}

    try:
        response = client.get_object("datasets", label_json_path)
        # existing_labels = json.loads(response.read().decode("utf-8"))
        response.close()
        response.release_conn()
        print(f"Existing {label_json_path} loaded.")
    except Exception:
        print(f"Creating new {label_json_path}.")

    image_objects = client.list_objects(
        "datasets", prefix=f"{dataset_name}/images/", recursive=True
    )
    image_tar_objects = [obj.object_name for obj in image_objects]

    query = """
        SELECT path, label, to_char(updated, 'YYYY-MM-DD') as updated
        FROM cifar10
        where is_processed = TRUE
    """
    rows += db.all(query)

    # Update metadata
    existing_labels["image_tars"] = image_tar_objects
    for row in rows:
        existing_labels["labels"].append(
            {"path": row.path, "label": row.label, "updated_date": row.updated}
        )

    # Upload updated JSON
    json_data = json.dumps(existing_labels, indent=2).encode("utf-8")
    client.put_object(
        "datasets",
        label_json_path,
        data=io.BytesIO(json_data),
        length=len(json_data),
        content_type="application/json",
    )

    # 5. Mark as processed in DB
    print("Marking rows as processed in DB...")
    paths = [row.path for row in rows]
    db.run("UPDATE cifar10 SET is_processed = TRUE WHERE path = ANY(%s)", (paths,))

    print(f"Success: Processed {len(rows)} images and updated labels.")


if __name__ == "__main__":
    dates = [
        "2026-03-22",
        "2026-03-23",
        "2026-03-24",
        "2026-03-25",
        "2026-03-26",
        "2026-03-27",
    ]

    version = 1
    for d in dates:
        create_dataset(d, "cifar10", f"v{version}")
        version += 1
