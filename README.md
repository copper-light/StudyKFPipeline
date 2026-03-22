# 비전기반 MLOps pipeline

### 1. data gather
* 디비에 이미지 경로, 데이터, 라벨링, 업데이트 일자가 기록되어 있음
* 게더 파이프라인은 날짜를 입력받아 해당 날짜 이하의 데이터중에 처리 되지 않은 데이터를 모두 가져와서 패스에 알맞게 오브젝트 스토리지에 {dataset_name}/{날짜}.gz.tar로 저장
* 라벨링 정보는 json 파일로 만들어서 labels_{버전}.json 으로 저장
    * 라벨링 정보에는 이전 데이터, 신규 데이터를 모두 포함하는 정보를 담음
    * 또한 라벨링 정보를 포함하는 일자의 ojbect 데이터를 목록으로 명시해야함


### 2. data processing pipeline
* 오브젝트 스토리지에 저장된 {dataset_name}/labels/label_v{버전}.json 파일을 다운로드 받아서 가져올 이미지 tar 목록을 확인
* tar 파일을 다운로드 받아 압축을 풀고 학습준비


### 3. train pipeline
