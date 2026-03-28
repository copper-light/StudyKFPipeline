import json

# =========================================================================
# 1. Katib 파이프라인(Experiment) 명세 (일반적인 Kubernetes YAML 문서 구조와 동일)
# =========================================================================
namespace = "handh"
experiment_name = "tuto-katib-random"

katib_experiment_manifest = {
    "apiVersion": "kubeflow.org/v1beta1",
    "kind": "Experiment",
    "metadata": {
        "name": experiment_name,
        "namespace": namespace
    },
    "spec": {
        # 최대로 실행할 Trial(하이퍼파라미터 학습 시도)의 수와 동시 실행 개수
        "parallelTrialCount": 2,
        "maxTrialCount": 6,
        "maxFailedTrialCount": 2,
        
        # 목표 수치 및 기준 Metric ('loss'를 최소화)
        "objective": {
            "type": "minimize",
            "goal": 0.001,
            "objectiveMetricName": "loss", # 학습 코드 안에서 'loss=값' 형태로 출력(표준출력)해야 Katib이 인식합니다.
        },
        
        # 최적화 알고리즘 (random, bayesianoptimization, grid 등)
        "algorithm": {
            "algorithmName": "random"
        },
        
        # 튜닝할 하이퍼파라미터 탐색 범위 세팅
        "parameters": [
            {
                "name": "learning_rate",
                "parameterType": "double",
                "feasibleSpace": {
                    "min": "0.01",
                    "max": "0.1"
                }
            },
            {
                "name": "batch_size",
                "parameterType": "int",
                "feasibleSpace": {
                    "min": "16",
                    "max": "64"
                }
            }
        ],
        
        # 실제로 학습이 이루어질 컨테이너(Pod / Job)의 템플릿
        "trialTemplate": {
            "primaryContainerName": "training-container",
            "trialParameters": [
                {
                    "name": "learningRate",
                    "description": "학습률",
                    "reference": "learning_rate"
                },
                {
                    "name": "batchSize",
                    "description": "배치 사이즈",
                    "reference": "batch_size"
                }
            ],
            "trialSpec": {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "training-container",
                                    "image": "python:3.11",
                                    # [데모용 학습 코드]
                                    # sys.argv로 Katib이 던져준 하이퍼파라미터를 받고, 
                                    # 임의의(random) loss 값을 뱉어내는 가상 학습 스크립트입니다.
                                    "command": [
                                        "python", "-c",
                                        "import random, time, sys; "
                                        "lr = float(sys.argv[1]); "
                                        "bs = int(sys.argv[2]); "
                                        "print(f'Training with lr={lr}, bs={bs}'); "
                                        "time.sleep(2); "
                                        "loss = random.uniform(0.01, 1.0) / lr; "
                                        "print(f'loss={loss}');" # 중요: Katib은 stdout 로그에서 "loss=" 를 스크래핑합니다.
                                    ],
                                    # Katib이 주입해주는 변수값들을 args로 매핑
                                    "args": [
                                        "${trialParameters.learningRate}",
                                        "${trialParameters.batchSize}"
                                    ]
                                }
                            ],
                            "restartPolicy": "Never"
                        }
                    }
                }
            }
        }
    }
}

if __name__ == "__main__":
    try:
        # kubeflow-katib SDK가 설치되어 있어야 합니다 (pip install kubeflow-katib)
        from kubeflow.katib import KatibClient
    except ImportError:
        print("[오류] kubeflow-katib 패키지가 설치되어 있지 않습니다.\n명령어: pip install kubeflow-katib")
        import sys
        sys.exit(1)

    try:
        # Katib 클라이언트 생성
        client = KatibClient()

        # 기존에 같은 이름의 실험이 있다면 삭제
        try:
            client.delete_experiment(name=experiment_name, namespace=namespace)
            print(f"Deleted existing experiment '{experiment_name}'")
        except:
            pass
        
        # Katib Experiment 배포 (Kubernetes에 적용)
        client.create_experiment(katib_experiment_manifest, namespace=namespace)
        print(f"Katib Experiment '{experiment_name}' successfully created in namespace '{namespace}'!")
        print("\nKubeflow의 'Experiments(AutoML)' 메뉴에서 실시간 진행 상황을 확인하세요.")
        
    except Exception as e:
        print(f"Failed to create Katib experiment: {e}")
