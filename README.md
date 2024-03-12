# L3_Classification_Deep Learning Model

**세 번째 요추 CT 데이터 분류 딥러닝 모델 개발 프로젝트**

### 논문
[L3_Classification_Thesis.pdf](https://github.com/yachae-sw/L3_Classification_DL/files/14548936/L3_Classification_Thesis.pdf)

| Lumbar Vertebrae 3 |
|---|
| <img src="https://github.com/yachae-sw/Muscle_Fat_Segmentation_DL/assets/93850398/c4c037e6-72d2-4394-a16f-92abd54063fd" width="350"> |


## 프로젝트 소개

- **프로젝트 목표 :** 복부 CT 영상에서 세 번째 요추 데이터 추출을 위한 최적화 알고리즘 및 분류 딥러닝 모델 개발
- **수행 기간 :** 2023.03 ~ 2022.07 (약 5개월)
- **팀 구성 :** 공동 연구(3명 진행)
- **담당역할 :**  CT 데이터 전처리 및  딥러닝 모델 훈련 담당


| No_L3_CT | L3_CT |
|---|---|
| <img src="https://github.com/yachae-sw/Muscle_Fat_Segmentation_DL/assets/93850398/2165054b-a004-4546-939e-f18e849580b7" width="300"> | <img src="https://github.com/yachae-sw/Muscle_Fat_Segmentation_DL/assets/93850398/4cb9cde8-c47d-43a5-b151-e4a80dd17cca" width="300"> |

## 데이터세트
- L3 슬라이스 검출을 위하여 전립선암 환자 104명, 방광암 환자 46명의 데이터가 사용되었다.
- 총 150명의 데이터 모두 횡단면 CT 영상으로 구성되었다.
- 본 데이터는 강릉아산병원에서 수집되었다
- 비뇨의학과 임상의가 ITK-SNAP v3.8 소프트웨어를 사용하여 L3 슬라이스를 수동으로 분할하였으며, 학습모델
의 출력 정보로 활용되었다.

| Dataset |
|---|
| ![classification_dataset](https://github.com/yachae-sw/L3_Classification_DL/assets/93850398/97363b5a-6241-4887-a899-2b63a4e6605e) |

## 프로젝트 진행 과정

1. 방광암 및 전립선암 환자 150명 CT data 및 Label data 전처리
    - 150명의 환자 CT 영상을 **DICOM 형식**에서 딥러닝 모델에 적용 가능한 형태로 변환하기 위해 Hounsfield Unit (HU) 값을 [-200 1500] 범위로 조절하고 **PNG 형식**으로 변환합니다.
    - 세 번째 요추 부분은 복부 내에서 소량으로 존재하기 때문에 **데이터 불균형** 문제가 발생합니다.
    이를 해결하기 위해 세 번째 요추 부분에 대해서 데이터 증강을 적용하고 클래스 가중치를 조절합니다.

2. 베이지안 최적화로 최적의 파라미터 값 추출
    - 최적의 데이터 비율과 클래스 가중치를 찾기 위해 **베이지안 최적화 방법**을 사용합니다.
    - 프로젝트에서 제안한 방법과 최적화를 적용하지 않은 모델의 결과를 비교하여 모델을 평가합니다.

| 연구 과정 |
|---|
| ![classification_절차](https://github.com/yachae-sw/L3_Classification_DL/assets/93850398/e95d6a7c-0400-469d-bb79-64ab1bbae433) |

## 딥러닝 모델

| ResNet50 |
|---|
| ![ResNet50](https://github.com/yachae-sw/L3_Classification_DL/assets/93850398/49d037c6-04ce-47ed-8d42-6a29b435e3e5) |


## 프로젝트 결과

컴퓨터 단층촬영 영상(CT)에서 3번 요추부 슬라이스 부분을 자동으로 추출한 후 지방 및 근육량을 95% 이상의 정확도로 예측하는 딥러닝 모델을 개발합니다.
- 기존의 딥러닝 모델과 최적화된 모델을 비교한 결과, 최적화된 모델이 향상된 성능을 보였습니다.
- 전신 CT 슬라이스에서 L3 부분을 1로 레이블을 지정하고, L3가 아닌 부분을 0으로 지정하여 이진 분류를 수행하였습니다.
- 150명의 예측된 세 번째 요추(L3)의 중간 부분과 실제 L3 중간 부분의 평균 슬라이스 오차는 0.68±1.26로 나타났습니다.
- 잘못 예측된 사례 중 대부분은 실제 중간 L3 슬라이스 이전 또는 이후의 슬라이스였습니다.
- 연구 결과는 데이터 증강을 통한 오버 샘플링과 클래스 가중치 조절을 통해 **데이터 불균형 문제를 효과적으로 해결**할 수 있음을 입증하였습니다.

![classification_Result](https://github.com/yachae-sw/L3_Classification_DL/assets/93850398/2730f80b-0272-458a-b54e-398d6a446624)

| 방법 | 훈련데이터 | 중간 L3 error |
|---|---|---|
| FCN(kanavati et al, 2018) | Frontal MIP images | 0.87±4.66 |
| VGG16 (Beharbi et al, 2017) | Frontal MIP images | 2.04±2.54 |
| VGG11 (Dabiri et al, 2020) | Axial images | 0.87±2.54 |
| ResNet50(최적화 수행 x) | Axial images | 1.68±1.43 |
| **ResNet50(제안된 방법)** | **Axial images** | **0.68±1.26** |


## 프로젝트 후기

- 프로젝트를 통해 제안한 최적화 방법의 접근 방식이 **다양한 분야에서 데이터 불균형 문제**를 해결하는데 유용함을 확인했습니다.
- CT 데이터를 다뤄 보면서 딥러닝 환경에 적용하는 방법을 배웠으며 앞으로 MRI, 심전도, 근전도와 같은 의료 데이터를 다뤄보고자 합니다.
- L3 슬라이스 자동 추출 딥러닝 모델에 관한 부분은 한국정보전자통신기술학회논문지에 작성된 논문을 게재했습니다.