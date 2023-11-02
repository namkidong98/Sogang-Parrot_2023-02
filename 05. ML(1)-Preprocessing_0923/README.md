## 머신 러닝의 종류
1. 지도 학습(supervised learning)    
   레이블 된 데이터로 학습   
   종류 : 분류(데이터를 기반으로 새로운 샘플의 범주형 클래스 레이블 예측), 회귀(연속적인 출력값에 대한 예측)   
2. 비지도 학습(unsupervised learning)   
   레이블 되지 않은 데이터로 학습   
   종류 : 군집화(같은 그룹 내 유사성), 차원 축소(고차원 데이터를 중요 정보 유지하며 압축, 잡음을 제거해 예측 성능을 높임)
3. 강화 학습(reinforcement learning)   
   특정 목표를 위해 최선의 전략을 선택하도록 학습
   시행 착오(Trial and Error), 지연 보상(Delayed Reward)

<br>

## 파라미터 vs 하이퍼파라미터
1. 파라미터(Parameter)   
   머신러닝 훈련 모델에 의해 요구되는 변수, 성능을 결정   
   데이터로부터 추정 또는 학습된다   
   개발자에 의해 수동으로 설정하지 않음      
2. 하이퍼파라미터(HyperParameter)
   최적의 훈련 모델을 구현하기 위해 모델에 설정하는 변수   
   개발자에 의해 수동으로 설정할 수 있음

<br>

## 머신러닝의 단계
1. 문제 정의 & 가설 설정
2. 데이터 수집
3. 데이터 전처리
4. 모델링 및 훈련
5. 모델 평가 & 인사이트 도출

<br>

## 데이터 전처리 : 데이터를 분석 및 처리에 적합한 형태로 만드는 과정
1. 전반적인 데이터 확인    
   label 값 확인, 필요없는 열 삭제(drop), value_counts()를 통해 자료형 등 확인   
   **시각화: 범주형 데이터(barplot을 주로 사용), 수치형 데이터(displot을 주로 사용)**     
   cf) 수치형 데이터는 주로 정규 분포에 근사해야 한다 --> log 변환이 필요   
2. 결측치 처리   
   결측치 확인(isnull().sum()), 삭제(dropna())   
   **값 대체 : 최빈값(범주형 데이터), 평균, 중앙값(수치형 데이터), 기본값 할당**   
3. 이상치(Outlier) 처리
   이상치(Outlier) : 일반적인 패턴에서 벗어나 극단적으로 크거나 작은 값   
   양이 적을 때는 제거하는 것이 일반적, 양이 많을 때는 이상치가 아닌 '특징이 있는' 데이터   
   Z값, IQR 기반으로 판단   
  ```python
  income_Q1 = df['log_income'].quantile(0.25) # 1사분위
  income_Q3 = df['log_income'].quantile(0.75) # 3사분위
  income_IQR = income_Q3 - income_Q1

  income_lower_bound = income_Q1 - 1.5 * income_IQR
  income_upper_bound = income_Q3 + 1.5 * income_IQR
  df = df[(df['log_income'] >= income_lower_bound) & (df['log_income'] <= income_upper_bound)]   
  ```
4. 중복된 값 처리   
   중복된 값의 개수: duplicated().sum() / 중복된 값 제거 : drop_duplicates(keep="first")   
5. 자료형 변환 : astype()   
6. 인코딩   
   **범주형 데이터에 대해서는 인코딩을 따로 해주어야 한다**   
   1) Label Encoding   
      범주가 순서 또는 순위인 경우   
      범주형 데이터의 문자열을 그대로 숫자형으로 변환   
   2) One-Hot Encoding   
      범주 간 관계가 독립적인 경우   
      feature값의 유형에 따라 새로운 feature(더미변수)를 추가하여 고유값에 해당하는 column에만 1을 부여   
      cf) label encoding이 되어 있어도 one-hot encoding을 하는 경우가 잦다
8. 데이터 스케일링
   데이터의 범위를 재정위 하는 것
   1) Standard Scaler : 평균과 분산을 이용하여 데이터를 정규화(표준 정규화)
   2) Robust Scaler : 중간값과 사분위값을 이용하여 데이터를 정규화
   3) Minmax Scaler : 최대값과 최소값을 이용하여 데이터를 정규화(0 ~ 1 사이로 재구성)
