# README

### Train Server
- 모델을 학습하는 서버
- gemma2 2b 기반으로 학습 (RTX 3060 12GB)
- main.py
  - func_llama3(): 모델 학습 수행 및 모델 업로드 수행
  - func_monitor_data(): 생성된 rationale이 있는지 webserver에 주기적 확인
  - main(): 위의 두 함수를 비동기로 실행하면서 monitor_data()가 데이터 다운로드 중에 있으면 func_llama3() 시작을 잠깐 미룸
- train_parser_LLM.py
  - list와 dict 데이터를 잘 파싱하기 위해 학습 
  - 여기서 학습된 모델은 rationale server에서 활용

### Rationale Server
- Rationale을 생성하는 서버
- 학습된 gemma2 2b 기반으로 테스트 (RTX 4060 8GB)
- main.py
  - func_llama3(): 데이터 생성 및 데이터 업로드 수행
  - func_monitor_model(): 생성된 모델이 있는지 webserver에 주기적 확인
  - main(): 위의 두 함수를 비동기로 실행하면서 func_monitor_model()가 모델 다운로드 중에 있으면 func_llama3() 시작을 잠깐 미룸
- src/get_questions.py
  - 외부에서 데이터를 가져오기 위한 함수들
  - 현재는 reddit에서만 가져오는 기능 활성

### Web Server
- Train Server와 Rationale Server 간 데이터 전송을 위한 서버
- FastAPI 기반으로 제작
- 추후 DB 기능 추가 예정

### 동작 방식
- localhost/upload-file/
  - 생성된 파일 업로드하는 링크
  - post 방식으로 업로드할 파일을 전송
  - Train model은 학습된 모델을 업로드
  - Rationale model은 생성된 rationale을 업로드
- localhost/download-file/{filename}
  - 업로드된 파일 다운로드하는 링크
  - filename을 넣으면 업로드된 파일을 다운로드 수행
  - Train model은 생성된 rationale을 다운로드
  - Rationale model은 학습된 모델을 다운로드
- localhost/status-file/{message}
  - 현재 서버에 존재하는 파일을 알려주는 페이지
  - 학습된 모델과 생성된 rationale 파일 이름 정보를 저장
  - 여기서 받은 정보를 가지고 download-file 페이지를 방문

1. Train Server: 모델 학습 -> Web server에 모델 업로드
   1. Web Server에 Rationale 정보가 있으면 비동기로 다운로드
   2. 다운로드된 Rationale 정보가 있으면 다음 epoch에 학습 데이터로 활용
2. Rationale Server: Rationale 생성 -> Web Server에 업로드
   1. Web Server에 학습된 모델 정보가 있으면 비동기로 다운로드
   2. 다운로드된 모델이 있으면 다음 데이터 생성에 활용
3. 상용 LLM을 활용해서 양질의 rationale을 확보하는 것은 추후 고려 중

### Parser LLM
- 답변을 파싱하는데 너무 에러가 많이 생기기 때문에 파싱을 잘 할수 있도록 학습한 LLM