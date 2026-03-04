# Project Rules

[환경]
- 버전(sketch 버전)별로 전용 conda 환경을 구축하여 구현
- torchrun 기반 PyTorch DDP, bf16 mixed precision
- 4GPU DDP (RTX 4090 * 4EA)

[sketch 폴더]
- 사용자가 구현하고자 하는 학습에 대한 스케치를 작성하는 폴더. (diagram.png, description.md)
- png, md 를 읽고 claude가 이해한 내용을 understanding.md 로 반드시 저장할 것.
- train 코드를 구현할 때, 반드시 참고.

[코딩 컨벤션]
- 모든 함수/클래스에 docstring 반드시 작성
- 모든 함수 인자와 반환값에 type hint 반드시 작성

[train 폴더]
- 학습 구현을 위한 모든 코드 보관
- 새로운 구현을 진행할때 마다 user_guide.md 파일을 업데이트하여, 코드 사용방법 정리

[config 폴더]
- 학습의 config 를 정의하는 yaml 파일만 보관
- config 로딩은 Hydra 사용

[checkpoints 폴더]
- 학습실행 log 보관, 학습 log 모니터링은 항상 Tensorboard
- 학습 코드가 자동으로 실행 시점 기준 "YYMMDD_HHMM" 폴더를 생성
- 폴더 안에 보관: tb logs, 실행에 사용된 config yaml 복사본, checkpoint model
- checkpoint 저장: best_model.pt (val loss 기준 lowest), epoch별 저장 간격은 config yaml에서 지정 (save_every_n_epochs)
- tb 이미지 로깅 간격은 checkpoint 저장 간격과 별도로 config yaml에서 지정 (log_image_every_n_epochs)
- tb log 에 반드시 train loss, validation loss 포함
- tb log 에 반드시 input images, output visualization 포함 (output depth map 과 gt 비교, reconstructed image, output token 과 gt token 을 tsne 로 비교 등)
  : 서로 다른 clip 에서 5개의 샘플을 학습 시작 시 고정(fixed) 선정
  : clip = physical AI 영상 데이터의 연속 시퀀스 (embodied AI 도메인 용어)
  : input images = 단일 프레임의 여러 이미지 (multi-view 등)
  : epoch 0 -> input images + output visualization 모두 저장, epoch N (config 지정 간격) -> output visualization 만 저장
