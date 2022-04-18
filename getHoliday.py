from sklearn.preprocessing import StandardScaler

import pandas as pd
import datetime


data = pd.read_csv('201901-202003.csv')

# 결측치 확인
# 모두 '세종'의 경우 발생
isnull_CARD = data[data['CARD_CCG_NM'].isnull()==True]
isnull_HOM = data[data['HOM_CCG_NM'].isnull()==True]
print('카드이용지역_시도 :', isnull_CARD['CARD_SIDO_NM'].unique())
print('거주지역_시도', isnull_HOM['HOM_SIDO_NM'].unique())

# CARD_CCG_NM, HOM_CCG_NM의 결측치 = '세종시'
data.fillna('세종시', inplace=True)


# Holiday column 추가
# 주말
def getDay(year, month, day):
    dow = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    return dow[datetime.date(year, month, day).weekday()]


day2019 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
day2020 = [31, 29, 31, 30, 31, 30, 31]

wknd = []

# 2019
for month in range(12):  # 1 ~ 12월
    w = []
    for day in range(day2019[month]):
        w.append(getDay(2019, month, day))
    wknd.append(w.count('Sat') + w.count('Sun'))

# 2020
for month in range(7):  # 1 ~ 7월
    w = []
    for day in range(day2020[month]):
        w.append(getDay(2020, month, day))
    wknd.append(w.count('Sat') + w.count('Sun'))

# 201901 ~ 202003
yymm = list(data['REG_YYMM'].unique())

# 202004 ~ 07 추가
yymm.extend([202004, 202005, 202006, 202007])


# 공휴일
hol = [1, 3, 1, 0, 1, 1, 0, 1, 2, 2, 0, 1,  # 2019
       3, 0, 0, 2, 1, 0, 0]                 # 2020

# 주말 + 공휴일
holidays = [x+y for x, y in zip(wknd, hol)]
hol_df = pd.DataFrame({'REG_YYMM': yymm, 'Holiday': holidays})

df = pd.merge(data, hol_df)


# CARD와 HOM 일치 여부
# T = 0, SIDO = 1, F = 2
df['region_diff'] = 0
df.loc[df['CARD_SIDO_NM'] != df['HOM_SIDO_NM'], 'region_diff'] = 2
df.loc[(df['CARD_CCG_NM'] != df['HOM_CCG_NM'])
       & (df['CARD_SIDO_NM'] == df['HOM_SIDO_NM']), 'region_diff'] = 1

# AGE 정수화
df['AGE_int'] = df['AGE'].str.split('s').str[0].astype(int)

df.drop(['AGE', 'HOM_SIDO_NM', 'HOM_CCG_NM', 'FLC'], axis=1, inplace=True)
df.rename(columns={'SEX_CTGO_CD': 'SEX', 'AGE_int': 'AGE'}, inplace=True)


std_scaler = StandardScaler()

monthDate = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2019
             31, 29, 31]                                      # 2020

df = df[['REG_YYMM', 'Holiday', 'AMT', 'CSTMR_CNT', 'CNT']]
df = df.groupby(['REG_YYMM', 'Holiday'])['AMT', 'CSTMR_CNT', 'CNT'].sum()
df = pd.DataFrame(df, columns=['AMT', 'CSTMR_CNT', 'CNT']).reset_index()

std = std_scaler.fit_transform(df[['AMT', 'CSTMR_CNT', 'CNT']])
std = pd.DataFrame(std, columns=['AMT', 'CSTMR_CNT', 'CNT'])

df['month'] = monthDate
df['rat_month'] = df['Holiday']/df['month']
df['rat_AMT'] = std['AMT']
df['rat_CNT'] = std['CNT']
df['rat_CST_CNT'] = std['CSTMR_CNT']

df.drop(['CNT', 'CSTMR_CNT', 'AMT', 'Holiday', 'month'], axis=1, inplace=True)
df.set_index('REG_YYMM', inplace=True)


def normalize(numeric_dataset):
    minVal = numeric_dataset.min(axis=0)
    maxVal = numeric_dataset.min(axis=0)
    ranges = maxVal - minVal
    matrix_normalized = (numeric_dataset - minVal)/ranges

    return matrix_normalized, ranges, minVal


nor, ranges, minVal = normalize(df)

# Holiday가 AMT에 영향을 미치지 않는 것으로 결론
corr = nor.corr(method='pearson')
