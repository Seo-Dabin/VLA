Physical Ai 이미지를 nuScences 뷰로 변환 (각 데이터셋의 카메라 모델, extrinsic, intrinsic 참고)
nuScence 뷰 이미지 5개를 input 으로 받아서 origin 데이터 표현을 학습하는 것이 목표.
카메라 모델과 파라미터를 모두 알기 때문에, inverse splat 기반으로 nuScence view 이미지를 input 으로 받아서 physical ai 뷰 (front wide, cross left/right) plane 으로 feature map 생성

feature map 을 중간 표현으로 3가지 task 를 순차적으로 학습
stage 1 : depth estimation. 학습 가속화를 위해 pretrained 된 모델의 output 을 라벨로 활용 (Depth-Anything-V2-Metric-Outdoor-Small-hf)
stage 2 : image reconstruction. Front_wide, Cross_left,right 의 원본 이미지를 활용하여 image reconstruction task 수행
stage 3 : Visual token decoder. Alpamayo-r1-10b 모델의 프로세서인 Qwen3-VL-2B-Instruct processor 를 거치면 각 이미지는 180개의 visual token 으로 압축됨. origin image 를 processor 를 거쳐서 visual token 라벨을 생성하고, 이를 재생성하는 것을 목표로 학습 진행. 여기서 processor (teacher model) 의 attention map 을 visual token decoder (student model) 의 attention map 이 모방하도록 하는 loss 도 함께 최적화.