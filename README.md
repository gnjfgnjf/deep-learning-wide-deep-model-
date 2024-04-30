# 도서추천 모델
![Untitled](https://github.com/gnjfgnjf/machinelearning_3/assets/156265958/54348ba8-af3a-4604-9345-43cb1666dd18)

**시기**

- 프로젝트 진행 기간 : 2024.03.07 ~ 2024.03.08

**요약**

- 고객들이 자신의 관심사와 선호도에 맞는 책을 빠르게 찾을 수 있도록 도움을 줄 수 있는 개인 맞춤형 도서 추천 모델을 설계하였습니다. wide-deep model을 사용하였고 데이터 전처리와 모델 최적화를 통해 MSE 13.78 의 값을 도출하였습니다.

**전처리**

1. test_df의 ‘Book-Author’ column 결측치 : NaN -> ‘Na’로 변환
2. Age : 카테고리형으로 분류 0~9살 -> 0 , 10~19살 -> 10 ,,, 100살 이상은 도서 추천을 해주더
라도 파악하기 힘든 나이라 생각하여 모두 묶어서 100으로 처리
= 새로운 column ‘fix_age’에 저장
3. 'Year-Of-Publication’ 에 -1 값이 너무 많아 'Publisher’로 group by 해준 후 같은 publisher를
갖고있는 값들의 평균으로 대체
3-1. 그 후 결측치를 갖게된 값들은 전체 평균값으로 대체
4. 연속형, 카테고리형 변수 분류 정의
연속형 변수 = 'fix_age', 'Year-Of-Publication’
카테고리형 변수 = 'User-ID', 'Book-ID', 'Location', 'Book-Title', 'Book-Author', 'Publisher’
5. 연속형 변수 결측치 처리, 표준화
결측치 = mean
표준화 = StandardScaler
6. 카테고리형 변수 전처리 : 상위 100개의 가장 빈번한 카테고리만 유지, 그 외 = 'Other’로 그룹화
= 모델 성능 최적화 -> 범주형 변수에 대한 차원 축소
= 자주 발생하지 않는 카테고리에 대한 과적합 위험 감소, 더 빈번한 카테고리에 집중&학습
7. 카테고리형 변수 LabelEncoder 사용
* feature 전처리 :
Wide Component 에서의 cross-product / Deep Component 에서의 embedding 을 사용했을 때의 결과값보다
사용하지 않은 결과값이 더 좋게 나와 해당 전처리는 하지 않았습니다.


![Untitled (1)](https://github.com/gnjfgnjf/machinelearning_3/assets/156265958/1cb9cec8-d1a5-40b0-ab2d-81b97f615f59)

![Untitled (2)](https://github.com/gnjfgnjf/machinelearning_3/assets/156265958/fcfc7feb-0994-4c9c-8fae-e00efb20aa48)

![Untitled (3)](https://github.com/gnjfgnjf/machinelearning_3/assets/156265958/c2e974fb-e540-44fb-ad71-febba4cba791)

![Untitled (4)](https://github.com/gnjfgnjf/machinelearning_3/assets/156265958/4bba4205-6af4-431d-bfca-beafdbe52f63)
