import pandas as pd

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
    ]
index = ["031", "059", "033", "045", "023"]
columns = ["종목명", "시가", "종가"]

df=pd.DataFrame(data=data, index=index, columns=columns)

print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
print("===========시가 1100원 이상인 행============")

print(df.loc["059"])
print(df.loc["033"])
print(df.loc["045"])

# print(df.)

print(df["종가"]["059"])
print(df["종가"]["033"])
print(df["종가"]["045"])

# filtered_df = df.query('시가 >= "1100"')
# filtered_df = df[df['시가'].astype(int) >= 1100]
# filtered_df = df[pd.to_numeric(df['시가']) >= 1100]
# filtered_df = df[df['시가'].apply(lambda x: int(x) >= 1100)]
# filtered_df = df.iloc[(df['시가'].astype(int) >= 1100).values]
# filter= df.loc[df['시가'].astype(int) >= 1100]
# filter= df.iloc[(df.iloc[:, 1].astype(int) >= 1100).values]
# filter= df.loc[df.iloc[:, 1].astype(int) >= 1100]
filter= df[df['시가']>='1100']
print(filter)
print("=====================================")


# ff = df.loc[df['시가'].astype(int) >= 1100]["종가"]
# ff= df.loc[df['시가'].astype(int) >= 1100, '종가']
# ff = df.query('시가 >= "1100"')['종가']
# ff= df[df['시가'].apply(lambda x: int(x) >= 1100)]["종가"]
# ff = df[pd.to_numeric(df['시가']) >= 1100]["종가"]
# ff= df[df['시가'].astype(int) >= 1100].iloc[:,2]
# ff = df[df['시가']>='1100']['종가']
ff = df[df['시가']>='1100'].iloc[:,2]

print(ff)
