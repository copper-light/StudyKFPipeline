import datetime
import hashlib
from kfp import dsl
from kfp import kubernetes
from kfp import compiler

# 1. 데이터 로드 및 전처리
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['minio', 'pandas', 'scikit-learn']
)
def load_and_preprocess(
    minio_endpoint: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    tar_object_name: str,
    preprocessed_data: dsl.Output[dsl.Dataset]
):
    import tarfile
    import os
    import pandas as pd
    from minio import Minio

    # MinIO 클라이언트 설정
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    
    # 1) MinIO에서 tar 파일 다운로드
    local_tar_path = '/tmp/dataset.tar.gz'
    print(f"Downloading {tar_object_name} from {bucket_name}...")
    client.fget_object(bucket_name, tar_object_name, local_tar_path)
    
    # 2) 압축 해제
    extracted_dir = '/tmp/extracted'
    os.makedirs(extracted_dir, exist_ok=True)
    with tarfile.open(local_tar_path, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)
        print(f"Extracted files: {os.listdir(extracted_dir)}")
        
    # 데모용 더미 데이터프레임
    df_processed = pd.DataFrame({'feature1': [10.5, 21.2, 33.1, 40.5], 'target': [0, 1, 0, 1]})
    
    # KFP Output(Dataset)에 결과 저장 
    df_processed.to_csv(preprocessed_data.path, index=False)
    print("Preprocessed data saved to Artifact path.")


# 2. 모델 학습 (TensorBoard PVC 모니터링 추가)
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'joblib', 'tensorboardX']
)
def train_model(
    preprocessed_data: dsl.Input[dsl.Dataset],
    n_estimators: int,
    max_depth: int,
    tensorboard_log_dir: str,
    run_name: str,
    model_output: dsl.Output[dsl.Model]
):
    import os
    import pandas as pd
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from tensorboardX import SummaryWriter
    
    # 1) 전처리 데이터 로드
    df = pd.read_csv(preprocessed_data.path)
    X = df[['feature1']]
    y = df['target']
    
    # 2) 텐서보드 디렉토리 설정 (고유 run_name을 하위 폴더명으로 사용하여 파이프라인 실험별로 텐서보드 격리 관리)
    experiment_log_dir = os.path.join(tensorboard_log_dir, run_name)
    os.makedirs(experiment_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_log_dir)
    print(f"TensorBoard logs will be written to PVC mount path: {experiment_log_dir}")
    
    # 3) 모델 학습
    print(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}")
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    
    # [텐서보드 더미 로그 기록 시뮬레이션]
    for epoch in range(10):
        # RandomForest 자체는 배치 학습이 아니지만 학습 진행율(Loss, Accuracy 등) 데모
        dummy_loss = np.random.random() * (10 - epoch)
        writer.add_scalar('Loss/train', dummy_loss, epoch)
    writer.close()
    
    # 4) 모델 아티팩트 저장
    joblib.dump(clf, model_output.path)
    
    # 5) 메타데이터 관리
    model_output.metadata['n_estimators'] = n_estimators
    model_output.metadata['max_depth'] = max_depth
    model_output.metadata['framework'] = 'scikit-learn'
    model_output.metadata['algorithm'] = 'RandomForest'
    model_output.metadata['tensorboard_log_dir'] = experiment_log_dir
    print("Model trained and safely metadata updated.")


# 3. 모델 평가
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def evaluate_model(
    preprocessed_data: dsl.Input[dsl.Dataset],
    model_input: dsl.Input[dsl.Model],
    eval_metrics: dsl.Output[dsl.Metrics]
):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    
    df = pd.read_csv(preprocessed_data.path)
    X = df[['feature1']]
    y = df['target']
    
    clf = joblib.load(model_input.path)
    
    preds = clf.predict(X)
    accuracy = accuracy_score(y, preds)
    
    eval_metrics.log_metric('accuracy', accuracy)
    print(f"Model evaluation accuracy: {accuracy}")


# 전체 파이프라인 정의
@dsl.pipeline(
    name='minio-ml-pipeline',
    description='MinIO Tar 데이터부터 모델 학습/평가, 텐서보드 PVC 모니터링 및 메타 관리를 지원하는 파이프라인'
)
def end_to_end_pipeline(
    run_name: str, # 파이프라인 내부에서도 고유 이름을 알 수 있도록 Argument로 추가
    minio_endpoint: str = 'minio-service.kubeflow.svc.cluster.local:9000',
    access_key: str = 'minio',
    secret_key: str = 'minio123',
    bucket_name: str = 'mlpipeline',
    tar_object_name: str = 'raw_data.tar.gz',
    n_estimators: int = 100,
    max_depth: int = 5,
    tensorboard_pvc_name: str = 'tensorboard-pvc', # 텐서보드가 연결된 Kubernetes PVC 이름
    tensorboard_mount_path: str = '/mnt/tensorboard_logs' # 컨테이너 내부 텐서보드 로그 마운트 경로
):
    # 컴포넌트 1: 데이터 다운로드 및 전처리
    preprocess_task = load_and_preprocess(
        minio_endpoint=minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
        tar_object_name=tar_object_name
    )
    
    # 컴포넌트 2: 모델 학습 (PVC 텐서보드 로깅)
    train_task = train_model(
        preprocessed_data=preprocess_task.outputs['preprocessed_data'],
        n_estimators=n_estimators,
        max_depth=max_depth,
        tensorboard_log_dir=tensorboard_mount_path,
        run_name=run_name
    )
    
    # KFP v2 kubernetes 모듈을 사용하여 train_task에 텐서보드용 PVC 볼륨 마운트
    kubernetes.mount_pvc(
        train_task,
        pvc_name=tensorboard_pvc_name,
        mount_path=tensorboard_mount_path
    )
    
    # 컴포넌트 3: 학습된 모델 평가
    evaluate_task = evaluate_model(
        preprocessed_data=preprocess_task.outputs['preprocessed_data'],
        model_input=train_task.outputs['model_output']
    )

if __name__ == '__main__':
    # [실험 추적] 날짜와 고유 해쉬값 생성
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_val = hashlib.sha256(current_time.encode()).hexdigest()[:8]
    
    # 이 이름을 토대로 파이프라인 실행 이름(Run Name)이 지정됨
    # 입력 파라미터(argument)로도 넘겨주어 TensorBoard 내부 로그 폴더 등으로 쓰이게 함
    run_name = f"ML-Run-{current_time}-{hash_val}"
    
    pipeline_root = f"s3://mlpipeline/artifacts/{run_name}"
    yaml_file = 'minio_pipeline.yaml'
    
    # YAML 컴파일
    compiler.Compiler().compile(
        pipeline_func=end_to_end_pipeline,
        package_path=yaml_file
    )
    
    print(f"✅ Pipeline compiled successfully: {yaml_file}")
    print(f"🔹 Generated Run Name: {run_name}")
    print(f"🔹 TensorBoard logs will be grouped by this run_name inside PVC: /mnt/tensorboard_logs/{run_name}")
    print("\n--- 실행 예시 Client 코드 ---")
    print(f'''
from kfp.client import Client
client = Client(host='http://<KUBEFLOW_ENDPOINT>')

client.create_run_from_pipeline_package(
    pipeline_file='{yaml_file}',
    arguments={{
        'run_name': '{run_name}',
        'tar_object_name': 'raw_data.tar.gz',
        'n_estimators': 150,
        'tensorboard_pvc_name': 'tensorboard-pvc',
        'tensorboard_mount_path': '/mnt/tensorboard_logs'
    }},
    run_name='{run_name}',
    pipeline_root='{pipeline_root}'
)
    ''')
