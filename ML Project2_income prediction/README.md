## 프로젝트명 : Korean Income Prediction

### 목표
동일한 사람을 일정 주기로 소득을 분석한 데이터인 Kor_income.csv의 데이터를 토대로 소득을 예측하는 모델을 만든다.


### 프로젝트 기간
2023.10.27 ~ 2023.11.08   
(전처리: 2023.10.27 ~ 2023.11.01 / ML: 2023.11.01 ~ 2023.11.08)

<br>

### 데이터셋 : Korea Income and Welfare
<img width="800" src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/cb52e564-22ca-446d-af9f-1a9ab963d52c">
https://www.kaggle.com/datasets/hongsean/korea-income-and-welfare

<br>

### Kor_income 데이터의 column에 대한 간단한 설명
<img width="1000" src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/4eeb7b11-715c-4d34-bbb2-0452c4b57c1f">


### <프로젝트 요약>
1. 데이터 전처리
     1) 중복값, 결측치 확인 및 처리
     2) Occupation, Company_size, Reason_none_worker의 결측치 및 형변환
     3) 나머지 column별 값 확인
     4) 이상치 제거 : marriage, religion, family_memeber
     5) label인 'income'의 이상치 처리 : IQR 사용, log변환
     6) 데이터 시각화 : income과 각 열의 barplot, 상관계수 히트맵
     7) Encoding : region, marriage, occupation, reason_none_worker열에 대한 one-hot encoding
2. 모델 선정과 학습 및 예측
     1) Linear Regression
     2) Decision Tree
     3) Random Forest
     4) Gradient Boosting
     5) KNN
3. 새로운 데이터에 대한 예측 및 제출
     1) 기존 데이터로 학습
     2) submission.csv에서 새로운 데이터 불러와서 전처리
     3) 모델 학습 및 임금 예측

<br>

### 1. 데이터 전처리
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/c3a74522-2094-4055-963f-98990c1a9c2b">

- Kor_income 데이터의 양식

- 2. Occupation, Company_size, Reason_none_worker 데이터가 보기에는 수치형 같은데, dtype이 object로 찍힌다
 
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/0d71a171-e6d1-4fe8-bf43-80f04fdf5250">

해결책: 각 컬럼에 대해 빈 공백을 제거하고 공백만 있는 데이터는 NaN으로 대체하고 fillna로 결측치를 변경한 후 int형으로 변환

<br>

- 4. marriage, religion, family_member에 섞인 99, 0과 같은 이상치를 적절히 처리한다

- 5. income을 IQR로 outlier를 처리하고 박스 플롯을 그려본 후 음수는 제거하고 income의 분포가 left-skewed하기 때문에 log변환을 통해 정규 분포의 형태로 바꿔준다
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/91f77a90-5c4b-4cca-a3b3-d784343e5ac6">
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/88193a2e-7442-4805-8ea8-0a1afc6fd72d">

<br><br>

- 6. income과 각각의 column들 사이의 관계성을 보기 위해 barplot을 사용하여 데이터를 시각화하였고, 상관계수를 히트맵으로 그렸다
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/0ef4ecac-bf55-42f4-93ab-3fec8f78fa4b">
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/763fd0a6-a2ff-45c9-b643-1967cf95e2ab">

family_memeber, year_born, education_level, company_size, occupation이 소득과 양의 상관관계를 갖고 있으며    
reason_none_worker과 gender가 소득과 음의 상관관계를 갖고 있는 것을 확인되며,    
id, year, wave, region, marriage, religion이 크게 상관관계가 없는 데이터로 확인된다.     

<br>

- 7. 범주형 데이터에 대해 원핫 인코딩을 수행한다
<img width=600 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/e6ae8505-8a6c-4e60-b78f-866dd1fe2cdf">

<br><br>

### 2. 모델 선정과 학습 및 예측
- 5가지 모델과 그리드 서치를 이용하여 RMSE를 최소화하는 모델을 찾는다
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/2dd681f1-feb8-4bf4-b58f-aa768d1e0e85">
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/74fe609d-abe2-4860-86d5-0a12ab379bf1">
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/bfaa10af-ead2-41ea-8740-5b1b9540aeec">
<img width=800 src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/857f2dcb-9c1e-4604-8187-adff83da8036">

- 그리드서치를 통해 하이퍼 파라미터를 조절한 XGBoost의 RMSE가 0.4392이고 그리드서치를 통해 하이퍼 파라미터를 조절한 RandomForest가 0.4395로 RMSE를 가장 최소화한 모델이다
- 그러나 XGBoost의 train score가 0.85, test score가 0.74인데 반해 RandomForest의 train score는 0.89, test score는 0.74이기 때문에 XGBoost가 overfitting의 염려가 적다고 판단하여 최종 모델로는 XGBoost를 선택했다.

<br>

### 프로젝트 발표 이후 생긴 질문
1. gender처럼 2개의 값만 있는 경우에도 one-hot encoding을 사용하나?     
     Answer: 그렇다. drop_first를 쓰면 어차피 1개 열만 남는다     
2. 범주가 어느 정도 있으면 처리하는게 좋은가? 예를 들어 이번 region은 범주가 7개인데 줄이는게 좋을까?     
     Answer: 7개 정도면 크게 줄일 필요가 없어 보인다     
3. get_dummies의 drop_first를 쓰는 것이 좋은 이유는 무엇인가?      
     Answer: 다중 공선성 문제를 해결할 수 있기 때문에 더 좋은 성능을 보인다.

<br>

### 프로젝트 발표 이후 발전시킬 부분
1. 오차분포 그래프를 그려보자
2. year과 year_born의 차를 통해 age column을 새로 만들어 보는 것도 시도해보자
3. test 데이터에 대해 이상치를 없애 버리지 않는 것이 나중에 모델의 일반화 성능 측면에서 좋다
4. 실제값, 예측값의 그래프를 그려보고, 떨어진 이상치들에 대해 다시 고민해보면 좋을 것 같다
5. (다른 조의 발표를 바탕으로) XGBoost는 Age를 최상위 피쳐로 본 예측, CatBoost는 가족 수를 최상위 피쳐로 본 예측이기 때문에 둘의 평균을 내는 ensemble을 쓰면 성능이 더 좋아질 것이다
6. 앙상블을 했을 때, 성능이 떨어졌다고 해서 안 좋아진게 아니다. 오히려 테스트 데이터에 대해서는 일반화 성능이 올라간 것이기 때문에 앙상블 기법을 사용하는 것을 추천한다
7. 전처리를 완성시키고 나서 모델을 돌려가면서 성능을 평가한다기 보다는 전처리를 하고 모델을 돌려보고 미진한 부분 개선하고 다시 모델 돌려보고 하는 식으로 한다.
8. IQR로 outlier를 처리할 때는 test 데이터는 남기고 train 데이터에 대해서 처리하는 것이다.
9. AutoML - NAS, CASH, Bayesian Optimization, Pycarcet library에 대해 공부해보자.
