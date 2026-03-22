from minio import Minio
import json
import tarfile

client = Minio(
    endpoint="localhost:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False,
)


def load(
    dataset_name: str = "cifar10", version: str = "v1", downalod_path: str = "tmp/"
):
    labels_manifest_path = f"{dataset_name}/labels/labels_{version}.json"

    try:
        image_tar_name = labels_manifest_path.split("/")[-1]
        client.fget_object(
            "datasets", labels_manifest_path, f"{downalod_path}/{image_tar_name}"
        )
        with open(f"tmp/{image_tar_name}", "r") as f:
            json_data = json.loads(f.read())
            image_tars = json_data["image_tars"]
            labels = json_data["labels"]

            for image_tar_path in image_tars:
                image_tar_name = image_tar_path.split("/")[-1]
                client.fget_object(
                    "datasets", image_tar_path, f"{downalod_path}/{image_tar_name}"
                )

                with tarfile.open(
                    f"{downalod_path}/{image_tar_name}", "r:gz"
                ) as tar_ref:
                    tar_ref.extractall(f"{downalod_path}/")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    load("cifar10", "v1", "tmp/")
