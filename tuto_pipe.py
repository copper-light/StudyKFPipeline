import os
from kfp import compiler, dsl
from kfp.dsl import Dataset, Model, Output, Input, HTML, Metrics
from typing import NamedTuple

@dsl.component(base_image="python:3.11", packages_to_install=["numpy"])
def create_dataset(dataset_url:str, dataset: Output[Dataset]) -> NamedTuple("Outputs", [("dataset_url", str)]):
    print("com1", dataset_url)
    with open(dataset.path, "w") as f:
        f.write("col1,col2,col3,col4")
        f.write("1,2,3,4")
        f.write("1,2,3,4")
        f.write("1,2,3,4")
        f.write("1,2,3,4")

    return (dataset_url,)

@dsl.component(base_image="python:3.11", packages_to_install=["numpy"])
def train(batch_size: int,
    num_epochs: int,
    learning_rate: float,
    dataset_url:str,
    dataset: Input[Dataset],
    ret: Output[HTML],
    matrics: Output[Metrics]):
    import numpy as np
    print("train", num_epochs, learning_rate, dataset.path)
    with open(dataset.path) as f:
        lines = f.readlines()

        for row in lines:
            print(row)

    with open(ret.path, "w") as f:
        f.write(
            "<HTML><p>Hellow world</p></HTML>"
        )

    matrics.log_metric("loss", 0.01)


@dsl.pipeline(name="tuto-pipeline", description = "hi I'm tuto pipe")
def run_pipeline(
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    dataset_url: str,
):
    task = create_dataset(dataset_url = dataset_url)
    loss = train(batch_size = batch_size, num_epochs = num_epochs, learning_rate = learning_rate, 
                 dataset_url = task.outputs["dataset_url"],
                 dataset = task.outputs["dataset"])


if __name__ == "__main__":
    from kfp.client import Client
    import datetime
    import sys
    
    yaml_file = os.path.basename(__file__).split()[-1] + ".yaml"
    compiler.Compiler().compile(
        pipeline_func = run_pipeline, package_path = yaml_file
    )
    print(f"Pipeline compiled successfully: {yaml_file}")

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run-{current_time}"

    try:
        client = Client()

        pipeline_name = "tuto-registered-pipeline"
        version_name = f"v-{current_time}"
        namespace = "handh"
        
        # 1. KFP 대시보드의 Pipelines 탭에 명시적으로 파이프라인 업로드 (초기 등록)
        try:
            print(f"Uploading pipeline '{pipeline_name}' to Kubeflow...")
            pipeline_info = client.upload_pipeline(
                pipeline_package_path=yaml_file,
                pipeline_name=pipeline_name,
                pipeline_version_name=version_name,
                namespace = namespace,
                description="This pipeline is registered in UI"
            )
            pipeline_id = pipeline_info.pipeline_id
            version_id = version_info.pipeline_version_id
            print(f"Pipeline successfully registered with ID: {pipeline_id}")
            
        except Exception as e:
            # 2. 이미 등록된 파이프라인이라면 버전(Version) 단위로 업데이트 관리
            print(f"Pipeline '{pipeline_name}' already exists. Attempting to fetch its ID...")
            pipeline_id = client.get_pipeline_id(pipeline_name)
            
            
            print(f"Uploading a new pipeline version: {version_name}")
            
            # 파이프라인의 하위 버전으로 YAML 파일 등록
            version_info = client.upload_pipeline_version(
                pipeline_package_path=yaml_file,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id
            )
            version_id = version_info.pipeline_version_id
            print(f"New version uploaded successfully with ID: {version_id}")

        # 3. Experiment 생성 또는 가져오기
        experiment = client.create_experiment(name="tuto-exper", namespace="handh")

        # 4. 등록된 Pipeline ID (및 특정 버전)으로 실험(Run) 실행
        run = client.run_pipeline(
            experiment_id=experiment.experiment_id,
            job_name=run_name,
            pipeline_id=pipeline_id,
            version_id=version_id, # 특정 버전을 명시적으로 지정
            params={
                "batch_size": 32,
                "num_epochs": 10,
                "learning_rate": 0.1,
                "dataset_url": "cifar10-v1" 
            }
        )

        final_run_info = client.wait_for_run_completion(
            run_id = run.run_id, timeout=3600
        )
        status = final_run_info.state
        print(f"Pipeline finished with status: {status}")
        if status == "SUCCEEDED":
            print("Pipeline Success!")
        else:
            print(f"[E] Pipeline Failed: {status}")
            sys.exit(1)

    except Exception as e:
        print(f"[E] Failed to create pipeline run: {e}") 
        sys.exit(1)