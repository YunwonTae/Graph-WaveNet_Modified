# Graph-WaveNet_Modified

This is the modified version of original Graph-WaveNet model(https://github.com/nnzhan/Graph-WaveNet). \
The additional incident and weather features are included on top of the original model.

<p align="center">
  <img width="350" height="400" src=./figure.png>
</p>

## Requirements
- python 3
- pytorch
- scipy
- numpy
- pandas
- pyaml
- pickle

## Data Preparation
  각 폴더 순서대로 파일경로만 설정하여 실행시켜주시면 됩니다.
### 1. create_routelist
  공통된 주요도로 찾기
### 2. create_dataframe
  공통된 주요도로를 바탕으로 속도 input파일 만들기 
### 3. create_incident
  돌발 전처리
### 4. weather
  기상 전처리 (시간이 상당소요 - 코드수정하여 시간을 줄여야함)
### 2-1. generate_dataframe_train_data
  속도 학습데이터 만들기
### 2-2. generate_incident_train_data
  돌발 학습데이터 만들기
### 2-3. generate_weather_train_data
  기상 학습데이터 만들기
### 2-4. generate_adj_mx
  인접행렬 만들기

## Experiment
 User needs to provide specific data path.
```
python3 train.py --adjtype doubletransition --addaptadj  --randomadj --incident true --weather true
```
