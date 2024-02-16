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
print("========================")
# print(df[0]) - error
# print(df["031"]) - error 
print(df["시가"]) # pandas의 기준은 열(columns)이다

print(df["종목명"][3])  # 아모레
print(df["종목명"]["045"])  # 아모레
print("========================")

# loc : 인덱스를 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출 ==> 인트loc 라고 생각
print(df.loc["031"])
# 종목명      삼성
# 시가     1000
# 종가     2000

# print(df.loc[3]) - error
print(df.iloc[3])
# 종목명     아모레
# 시가     3500
# 종가     6000
print(df.iloc[3][2])
# 6000
# print(df.loc["023"])    =  print(df.iloc[-1])   =   print(df.iloc[4])
print("========================")
# print(df.loc["045"].loc["시가"])    # 3500
# print(df.loc["045"].iloc[1])    # 3500
# print(df.iloc[3].loc["시가"])    # 3500
# print(df.iloc[3].iloc[1])    # 3500
# 전부 가능
print(df.loc["045"][1]) #되는데 경고뜸  
print(df.iloc[3][1])    #되는데 경고뜸

print(df.loc["045"]["시가"])    # 3500
print(df.iloc[3]["시가"])   # 3500

print(df.loc["045" , "시가"])   # 3500
print(df.iloc[3,1]) # 3500
print("========================")
print(df.iloc[3:5, 1])
print(df.iloc[[3,4], 1])
# print(df.iloc[3:5,"시가"]) - error
# print(df.iloc[[3,4], "시가"]) - error
print("========================")

print(df.loc[["045","023"],"시가"])
print(df.loc["045":"023","시가"])
# print(df.loc[3:5, "시가"]) - error
# print(df.loc[["045","023"],1]) - error
# print(df.loc["045":"023",1]) - error

