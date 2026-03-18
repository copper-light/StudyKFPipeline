import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, component

# 1. 첫 번째 컴포넌트: 데이터 준비 (Data Preparation)
@component(packages_to_install=["pandas", "scikit-learn"])
def prepare_data(
    output_dataset: Output[Dataset]
):
    import pandas as pd
    from sklearn.datasets import load_iris
    
    # 예제 데이터 로드 (Iris 데이터셋)
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # 다음 컴포넌트에서 사용할 수 있도록 데이터를 CSV 형식으로 저장
    df.to_csv(output_dataset.path, index=False)
    print(f"데이터가 다음 경로에 저장되었습니다: {output_dataset.path}")

# 2. 두 번째 컴포넌트: 모델 학습 (Model Training)
@component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def train_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model]
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # 이전 컴포넌트에서 전달받은 데이터 읽기
    df = pd.read_csv(input_dataset.path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 랜덤 포레스트 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 학습된 모델을 저장하여 다음 컴포넌트로 전달
    joblib.dump(model, output_model.path)
    print(f"학습된 모델이 저장되었습니다: {output_model.path}")

# 3. 세 번째 컴포넌트: 모델 평가 (Model Evaluation)
@component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def evaluate_model(
    input_dataset: Input[Dataset],
    input_model: Input[Model],
    metrics: Output[Metrics]
):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    
    # 데이터와 학습된 모델 로드
    df = pd.read_csv(input_dataset.path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    model = joblib.load(input_model.path)
    
    # 예측 수행 및 정확도 계산
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # 수집된 지표를 Kubeflow UI에 표시하기 위해 로깅
    metrics.log_metric("accuracy", accuracy)
    print(f"모델 정확도 (Accuracy): {accuracy:.4f}")

# 파이프라인 정의
@dsl.pipeline(
    name="three-component-ml-pipeline",
    description="3개의 컴포넌트로 구성된 간단한 파이프라인: 데이터 준비 -> 학습 -> 평가"
)
def simple_ml_pipeline():
    # 첫 번째 작업: 데이터 셋업
    prep_task = prepare_data()
    
    # 두 번째 작업: 데이터 준비 작업의 출력을 입력으로 받아 모델 학습
    train_task = train_model(
        input_dataset=prep_task.outputs['output_dataset']
    )
    
    # 세 번째 작업: 데이터와 학습된 모델을 입력으로 받아 평가
    eval_task = evaluate_model(
        input_dataset=prep_task.outputs['output_dataset'],
        input_model=train_task.outputs['output_model']
    )

if __name__ == "__main__":
    # 파이프라인을 실행 가능한 YAML 파일로 컴파일
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=simple_ml_pipeline,
        package_path="pipeline_tutorial.yaml"
    )
    print("파이프라인 컴파일이 완료되었습니다: pipeline_tutorial.yaml")
