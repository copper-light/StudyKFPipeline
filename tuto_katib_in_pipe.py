import os
from kfp import dsl, compiler
from typing import NamedTuple

# ========================================================
# 1. Pipeline 내에서 Katib을 띄우고 끝날 때까지 기다리는 컴포넌트
# ========================================================
@dsl.component(
    base_image="python:3.11", 
    packages_to_install=["kubeflow-katib==0.16.0", "kubernetes"] # 파이프라인 컨테이너 내부에 Katib SDK 설치
)
def run_katib_experiment(
    experiment_name: str, 
    namespace: str,
    max_trial_count: int
) -> NamedTuple("BestHPOptions", [("learning_rate", float), ("batch_size", int)]):
    from kubeflow.katib import KatibClient
    import time
    import json
    
    client = KatibClient()
    
    # 이전에 동일한 이름의 Experiment가 남아있다면 삭제
    try:
        client.delete_experiment(name=experiment_name, namespace=namespace)
        time.sleep(5)
    except:
        pass

    # Katib YAML (Python Dict 형태) 정의
    katib_experiment_manifest = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": experiment_name, "namespace": namespace},
        "spec": {
            "parallelTrialCount": 1,
            "maxTrialCount": max_trial_count,
            "maxFailedTrialCount": 1,
            "objective": {
                "type": "minimize",
                "goal": 0.001,
                "objectiveMetricName": "loss",
            },
            "algorithm": {"algorithmName": "random"},
            "parameters": [
                {"name": "learning_rate", "parameterType": "double", "feasibleSpace": {"min": "0.01", "max": "0.1"}},
                {"name": "batch_size", "parameterType": "int", "feasibleSpace": {"min": "16", "max": "64"}}
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {"name": "learningRate", "reference": "learning_rate"},
                    {"name": "batchSize", "reference": "batch_size"}
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "training-container",
                                    "image": "python:3.11",
                                    # 무작위로 loss를 생성해서 Katib이 잡을 수 있도록 stdout에 출력하는 데모 스크립트
                                    "command": [
                                        "python", "-c",
                                        "import random, sys; "
                                        "lr = float(sys.argv[1]); "
                                        "loss = random.uniform(0.01, 0.1) / lr; "
                                        "print(f'loss={loss}');"
                                    ],
                                    "args": ["${trialParameters.learningRate}", "${trialParameters.batchSize}"]
                                }],
                                "restartPolicy": "Never"
                            }
                        }
                    }
                }
            }
        }
    }
    
    print(f"Creating Katib Experiment: {experiment_name}")
    client.create_experiment(katib_experiment_manifest, namespace=namespace)
    
    # AutoML이 끝날 때까지 주기적으로 폴링(대기)
    print("Waiting for Katib Experiment to complete... (This will take a few minutes)")
    while True:
        try:
            exp = client.get_experiment(name=experiment_name, namespace=namespace)
            status = exp.get("status", {})
            conditions = status.get("conditions", [])
            
            if conditions:
                # 상태 조건리스트 중 가장 마지막(최신) 상태 확인
                last_condition_type = conditions[-1]["type"]
                if last_condition_type == "Succeeded":
                    print("Katib Experiment Succeeded!")
                    break
                elif last_condition_type in ["Failed", "Killed"]:
                    raise RuntimeError("Katib experiment Failed or was Killed.")
        except Exception as e:
            pass
            
        time.sleep(10) # 10초에 한 번씩 상태 확인
        
    # 완료 시 최적 하이퍼파라미터 추출
    exp = client.get_experiment(name=experiment_name, namespace=namespace)
    best_trial = exp["status"].get("currentOptimalTrial", {})
    best_params = best_trial.get("parameterAssignments", [])
    
    best_lr = 0.01
    best_bs = 32
    
    for p in best_params:
        if p["name"] == "learning_rate":
            best_lr = float(p["value"])
        elif p["name"] == "batch_size":
            best_bs = int(p["value"])
            
    print(f"=====================================")
    print(f"Best Hyperparameters Found!")
    print(f"LR: {best_lr}, Batch Size: {best_bs}")
    print(f"=====================================")
    
    from collections import namedtuple
    outputs = namedtuple("BestHPOptions", ["learning_rate", "batch_size"])
    return outputs(best_lr, best_bs)

# ========================================================
# 2. 위에서 추출한 최적의 하이퍼파라미터로 "진짜" 모델을 학습시키는 컴포넌트
# ========================================================
@dsl.component(base_image="python:3.11")
def train_best_model(learning_rate: float, batch_size: int):
    print("==================================================")
    print(f"Training FINAL Production Model with Katib's Best parameters")
    print(f"-> Learning Rate: {learning_rate}")
    print(f"-> Batch Size: {batch_size}")
    print("==================================================")
    # 진짜 머신러닝 학습 로직 구현 (Keras, PyTorch 등...)


# ========================================================
# 3. 위 둘을 이어주는 KFP 파이프라인
# ========================================================
@dsl.pipeline(name="kfp-katib-integration", description="파이프라인 안에서 Katib 돌리기")
def kfp_with_katib_pipeline(katib_exp_name: str = "pipeline-katib-demo", namespace: str = "handh"):
    
    # 1. 하이퍼파라미터 튜닝 실행 (완료 시까지 파이프라인 정지/블로킹됨)
    hpo_task = run_katib_experiment(
        experiment_name=katib_exp_name,
        namespace=namespace,
        max_trial_count=3 # 데모를 위해 3번만 시도
    )
    
    # 2. 최고 성능의 파라미터로 최종 모델 훈련
    final_model_task = train_best_model(
        learning_rate=hpo_task.outputs["learning_rate"],
        batch_size=hpo_task.outputs["batch_size"]
    )


if __name__ == "__main__":
    from kfp.client import Client
    import datetime
    
    yaml_file = "katib_integration_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=kfp_with_katib_pipeline, 
        package_path=yaml_file
    )
    
    client = Client()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"katib-pipeline-run-{current_time}"
    
    try:
        pipeline_info = client.upload_pipeline(
            pipeline_package_path=yaml_file,
            pipeline_name="KFP-Katib-Integration"
        )
        pipeline_id = pipeline_info.pipeline_id
    except:
        pipeline_id = client.get_pipeline_id("KFP-Katib-Integration")

    experiment = client.create_experiment(name="tuto-exper", namespace="handh")
    
    client.run_pipeline(
        experiment_id=experiment.experiment_id,
        job_name=run_name,
        pipeline_id=pipeline_id,
        params={
            "katib_exp_name": "pipeline-katib-" + current_time,
            "namespace": "handh"
        }
    )
    print("✅ 파이프라인 & Katib 통합 실험이 KFP에 성공적으로 예약되었습니다!")
