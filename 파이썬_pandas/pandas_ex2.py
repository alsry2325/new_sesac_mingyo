import pandas as pd
from tabulate import tabulate


df = pd.read_csv("파이썬_pandas/seoul_park.csv")

#최대 컬럼 폭 디스플레이 옵션
# pd.set_option('display.max_rows', None)

#디폴트로는 5개만 출력 인자값을 넣으면 그숫자만큼 출력함
#상위 행 일부를 데이터프레임으로 반환
print(df.head())
'''
  날짜 공휴일     날씨   유료합계     어른  청소년  어린이  외국인   단체   무료합계     총계
0  2016-01-01   O  구름 조금  3,359  2,799  141  419   47    0  1,023  4,382
1  2016-01-02   O  구름 많음  5,173  4,370  203  600  100  111  2,092  7,265
2  2016-01-03   O  구름 많음  3,008  2,571  128  309   91    0  1,549  4,557
3  2016-01-04   X  구름 많음    890    602  NaN  235   51  223    800  1,690
4  2016-01-05   X  구름 많음    416    319   35   62   43   47    840  1,256
'''
#하위 행 일부를 데이터프레임으로 변환 
print(df.tail())
'''
날짜 공휴일     날씨   유료합계     어른  청소년  어린이 외국인   단체   무료합계     총계
1081  2019-03-27   X  구름 많음    504    464   10   30  21    -    613  1,117
1082  2019-03-28   X  구름 많음    761    687   46   28  35  108    904  1,665
1083  2019-03-29   X  구름 조금  1,644  1,447  120   77  14  188  1,226  2,870
1084  2019-03-30   O     흐림  1,539  1,326   44  169  29  115    913  2,452
1085  2019-03-31   O  구름 조금  3,061  2,563  111  387  53    -  1,357  4,418
'''
#데이터의 정보 확인  인덱스개수 범위, 컬럼의 개수,컬럼의 이름, 각 컬럼의 데이터 개수 ,데이터 타입
print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1086 entries, 0 to 1085
Data columns (total 11 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   날짜      1086 non-null   object
 1   공휴일     1086 non-null   object
 2   날씨      946 non-null    object
 3   유료합계    1086 non-null   object
 4   어른      1086 non-null   object
 5   청소년     1081 non-null   object
 6   어린이     1086 non-null   object
 7   외국인     1086 non-null   object
 8   단체      1086 non-null   object
 9   무료합계    1086 non-null   object
 10  총계      1086 non-null   object
'''

#로우와 컬럼의 개수를 투플 형태로 반환
print(df.shape)
'''
dtypes: object(11)
memory usage: 93.5+ KB
None
(1086, 11)
'''

#컬럼의 이름들
print(df.columns)
'''
Index(['날짜', '공휴일', '날씨', '유료합계', '어른', '청소년', '어린이', '외국인', '단체', '무료합계',
       '총계'],
      dtype='object')
'''
# #각 컬럼의 데이터 타입 정보
print(df.dtypes)
'''
날짜      object
공휴일     object
날씨      object
유료합계    object
어른      object
청소년     object
어린이     object
외국인     object
단체      object
무료합계    object
총계      object
'''

#특정 컬럼 추출하기
# print(df["어른"]) 
'''
dtype: object
0       2,799
1       4,370
2       2,571
3         602
4         319
        ...
1081      464
1082      687
1083    1,447
1084    1,326
1085    2,563
Name: 어른, Length: 1086, dtype: object
'''

print(df.dtypes) # 각컬럼의 타입에 대한 정보

serise = df["날씨"]
print(serise)
'''
0       구름 조금
1       구름 많음
2       구름 많음
3       구름 많음
4       구름 많음
        ...
1081    구름 많음
1082    구름 많음
1083    구름 조금
1084       흐림
1085    구름 조금
Name: 날씨, Length: 1086, dtype: object
'''
result1 =   df["어른"]   #이건 시리즈 타입

'''
0       2,799
1       4,370
2       2,571
3         602
4         319
        ...
1081      464
1082      687
1083    1,447
1084    1,326
1085    2,563
'''
#여러 특정 컬럼 추출하기
result2 = df[["어른","외국인"]]     # 데이터 타입
'''
Name: 어른, Length: 1086, dtype: object
         어른  외국인
0     2,799   47
1     4,370  100
2     2,571   91
3       602   51
4       319   43
...     ...  ...
1081    464   21
1082    687   35
1083  1,447   14
1084  1,326   29
1085  2,563   53
'''
result3  = df[["어른"]]  # 이건 데이터 타입
print(type(result1))# <class 'pandas.core.series.Series'>
print(type(result3))# <class 'pandas.core.frame.DataFrame'>
print(result1)
print(result2)
print(result3)

#데이터 빈도 세는법 :value_count()  카테고리별 갯수세는 메소드
'''
[1086 rows x 1 columns]
날씨
구름 많음    277
구름 조금    236
맑음       222
비        101
흐림       100
눈          6
눈/비        4
Name: count, dtype: int64
'''
count = df["날씨"].value_counts()
print(count)

print(df["공휴일"].value_counts())  #결과는 series형태로 반환이름은count
#전체df에서 공휴일 컬럼을 가져와서 , 해당 컬럼에 대한 값의 빈도수를 구해라
'''
날씨
구름 많음    277
구름 조금    236
구름 조금    236
맑음       222
비        101
흐림       100
눈          6
눈/비        4
Name: count, dtype: int64
'''

columns = [
            '유료합계',"어른","청소년","어린이",
            "외국인","단체","무료합계","총계"
           ]

for column in columns:
    #str 들어오는 인자값을 문자열로 바꿈  replace 0,1로 나누는 함수 
    df[column] = df[column].str.replace(",","") 
    df[column] = df[column].str.replace("-","0") 
    
print(df[column])
# #데이터 타입 변환하기
df["어른"] = df["어른"].astype(int)
print(df["어른"])
df.info()
# #숫자형 타입으로 바꾸는법
df["유료합계"] = pd.to_numeric(df["유료합계"])
print(df["유료합계"] )

#전체를 숫자형타입으로 바꾸기
for column in columns:
    df[column] = pd.to_numeric(df[column])

df.info()
df[column]
#날짜형 타입으로 변환
df["날짜"] = pd.to_datetime(df["날짜"])

df.info()
print(df["날짜"])
#
df["연"] = df["날짜"].dt.year  #날짜에서 새로운 키값 생성
df["월"] = df["날짜"].dt.month
df["일"] = df["날짜"].dt.day
df["요일"] = df["날짜"].dt.dayofweek
print(tabulate(df.head(), headers='keys', tablefmt='psql'))

#시리즈 연산
df['매출'] = df['어른'] + df['청소년'] + df['어린이']
print(df['매출'])

df2 = pd.DataFrame({
    "이름" : ["철수", "길동", "영희"],
    "국어" : [75, 68, 85],
    "수학" : [80, 95, 60],
    "영어" : [85, 100, 98]
})
df2['사회'] = [70, 65, 55]
df2['평균점수'] =  (df2['국어'] + df2['수학'] + df2['영어'] + df2['사회']) / 4
df2['Example'] = 100
print(df2)

#데이터 값 매핑하여 시리즈 변환하기 map()

week = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}

df["요일"]  =  df["요일"].map(week)

print(tabulate(df.head(), headers='keys', tablefmt='psql'))


#데이터 값 함수 적용하여 변환하기

# def weather(e):
#     if e == "눈" or e == "비":
#         return "눈/비"
#     else:
#         return e

print(df["날씨"].value_counts()) # 호출전 눈이랑 비 따로 

#람다 매개변수: 조건 참 if  else 거짓일때 
df["날씨"] = df["날씨"].apply(lambda e : "눈/비" if e == "눈" or e == "비" else e)

#함수로 호출
# df["날씨"] = df["날씨"].apply(weather)

print(df["날씨"].value_counts())    # 호출 후 눈/바 합쳐짐 

#날씨별 총계의 평균
aggregate = df.groupby("날씨")["총계"].mean()
print(aggregate)
'''
날씨
구름 많음    6234.844765
구름 조금    7409.122881
눈/비      3038.603604
맑음       7756.225225
흐림       6056.450000
Name: 총계, dtype: float64
'''
aggregate = df.groupby(["날씨","공휴일"])["총계"].mean()
print(aggregate)
'''
날씨     공휴일
구름 많음  O      12994.262500
       X       3489.903553
구름 조금  O      14797.581081
       X       4034.148148
눈/비    O       4239.027778
       X       2462.400000
맑음     O      14658.260274
       X       4374.691275
흐림     O       9374.957447
       X       3113.622642
Name: 총계, dtype: float64
'''
#전체 컬럼에  대해서 집계
grouped = df.groupby("날씨").mean(numeric_only=True) # mean(numeric_only=True) 숫자형 정수만 평균을 내라
print(tabulate(grouped, headers='keys', tablefmt='psql'))

'''데이터프레임으로 
----------+------------+---------+----------+----------+----------+----------+------------+---------+---------+---------+---------+---------+
| 날씨      |   유료합계 |    어른 |   청소년 |   어린이 |   외국인 |     단체 |   무료합계 |    총계 |      연 |      월 |      일 |    매출 |
|-----------+------------+---------+----------+----------+----------+----------+------------+---------+---------+---------+---------+---------|
| 구름 많음 |    4278.68 | 3362.7  |  401.698 |  506.801 |  76.0433 |  819.426 |   1956.17  | 6234.84 | 2017.05 | 6.59567 | 16.1119 | 4253.69 |
| 구름 조금 |    5119.97 | 4012.06 |  485.54  |  587.555 |  73.3898 |  883.623 |   2289.15  | 7409.12 | 2017.18 | 6.28814 | 15.5551 | 5087.89 |
| 눈/비     |    2124.26 | 1411.9  |  282.766 |  421.099 |  47.7207 |  643.919 |    914.342 | 3038.6  | 2017.17 | 7.06306 | 15.8919 | 2115.77 |
| 맑음      |    5271.69 | 4051.32 |  641.109 |  575.518 |  71.6847 | 1127.62  |   2484.53  | 7756.23 | 2017.38 | 6.45045 | 15.8288 | 5287.88 |
| 흐림      |    4206.08 | 3452.79 |  208.65  |  542.18  |  78.81   |  547.39  |   1850.37  | 6056.45 | 2017.05 | 6.95    | 14.19   | 4203.62 |
'''
#'numpy.float64' object is not iterable 
# print(tabulate(grouped["어른"], headers='keys', tablefmt='psql')) 

print(grouped["어른"]) 