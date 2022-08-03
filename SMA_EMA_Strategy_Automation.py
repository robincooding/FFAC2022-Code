import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False # 한글 폰트를 위함


# ## 상장기업 데이터프레임 만들기 (코드, 이름)
def read_krx_code():
        """KRX로부터 상장기업 목록 파일을 읽어와서 데이터프레임으로 반환"""
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method='\
            'download&searchType=13'
        krx = pd.read_html(url, header=0)[0]
        krx = krx[['종목코드', '회사명']]
        krx = krx.rename(columns={'종목코드': 'code', '회사명': 'company'})
        krx.code = krx.code.map('{:06d}'.format)
        return krx


code_df = read_krx_code()


# ## 네이버 금융에서 시세 읽어오는 함수 만들기
from bs4 import BeautifulSoup
import requests
import ast

def read_naver(code, company):
        """네이버에서 최근 약 10개년 주식 시세를 읽어서 데이터프레임으로 반환"""
        try:
            url = 'https://api.finance.naver.com/siseJson.naver'
            params = {
                'symbol' : code, # 입력한 종목
                'requestType' : 0,
                'count' : 2500, # 2500일치 데이터(약 10년)
                'startTime' : datetime.today(), # 오늘부터 10년
                'timeframe' : 'day' # 일단위 데이터
            }
            res = requests.get(url, params = params)
            data = ast.literal_eval(res.text.strip()) # 공백 문자 제거 후 문자열을 리스트로 바꿔줌
            df = pd.DataFrame(data, columns = data[0]).drop(0) # 리스트를 데이터프레임으로 변환하고 첫 행을 열로 설정
            df = df.drop('외국인소진율', axis = 1) # 사용하지 않는 열 삭제
            df = df.rename(columns={'날짜':'date','종가':'close','시가':'open','고가':'high',
                                    '저가':'low','거래량':'volume'})
            df = df[['date', 'close']].set_index('date') # 필요한 정보만 빼서 재구성
            df['close'] = df['close'].astype('int') # 자료형 변환
            df.index = pd.to_datetime(df.index) # Datetime index로 변환
        except Exception as e:
            print('Exception occured :', str(e))
            return None
        return df


# ## 사용자 입력 받기
company_name = input("조회할 종목을 입력하세요: ") # 문제 : 정확한 종목명을 입력받지 못했을 때 인식할 방법

# **입력받은 종목의 코드를 반환하는 작업**
code = code_df.loc[code_df["company"] == company_name].code.values[0]

# **사용자가 입력한 종목에 대한 시세 데이터를 받아와 데이터프레임으로 변환**
df = read_naver(code, company_name)

# ## 분석 작업에 필요한 데이터프레임으로 변환
df.sort_index(inplace = True) # 데이터프레임 날짜 오름차순 정렬

# 장/단기 SMA/EMA값 설정하기
sma_s = 50
sma_l = 200
ema_s = 50
ema_l = 200

df["SMA_S"] = df.close.rolling(sma_s).mean() # 단기 SMA
df["SMA_L"] = df.close.rolling(sma_l).mean() # 장기 SMA
df["EMA_S"] = df.close.ewm(span = ema_s, min_periods = ema_s).mean() # 단기 EMA
df["EMA_L"] = df.close.ewm(span = ema_l, min_periods = ema_l).mean() # 장기 EMA
df.dropna(inplace = True)

# 전략에 따라 매수와 매도 신호를 표시하는 칼럼 추가하기
df["position_sma"] = np.where((df["SMA_S"].shift(1) < df["SMA_L"].shift(1)) & (df["SMA_S"] > df["SMA_L"]), 1, 0) # 단기 SMA가 장기SMA 위로 올라가면 매수 신호
df["position_sma"] = np.where((df["SMA_S"].shift(1) > df["SMA_L"].shift(1)) & (df["SMA_S"] < df["SMA_L"]), -1, df.position_sma) # 아래로 내려가면 매도 신호
df["position_ema"] = np.where((df["EMA_S"].shift(1) < df["EMA_L"].shift(1)) & (df["EMA_S"] > df["EMA_L"]), 1, 0) # 단기 EMA가 장기EMA 위로 올라가면 매수 신호
df["position_ema"] = np.where((df["EMA_S"].shift(1) > df["EMA_L"].shift(1)) & (df["EMA_S"] < df["EMA_L"]), -1, df.position_ema) # 아래로 내려가면 매도 신호
df["position_smaema"] = np.where((df["EMA_S"].shift(1) < df["SMA_S"].shift(1)) & (df["EMA_S"] > df["SMA_S"]), 1, 0) # EMA가 SMA 위로 올라가면 매수 신호
df["position_smaema"] = np.where((df["EMA_S"].shift(1) > df["SMA_S"].shift(1)) & (df["EMA_S"] < df["SMA_S"]), -1, df.position_smaema) # 아래로 내려가면 매도 신호

# 매수와 매도를 나타내는 조건 생성하기 (loc를 통한 접근 위함)
position_sma_buy = (df.position_sma == 1) # 매수 신호일때만 체크
position_sma_sell = (df.position_sma == -1) # 매도 신호일때만 체크
position_ema_buy = (df.position_ema == 1) # 매수 신호일때만 체크
position_ema_sell = (df.position_ema == -1) # 매도 신호일때만 체크
position_smaema_buy = (df.position_smaema == 1) # 매수 신호일때만 체크
position_smaema_sell = (df.position_smaema == -1) # 매도 신호일때만 체크

# SMA 전략 시각화하기
df.loc["03-2019":, ["close", "SMA_S", "SMA_L"]].plot(figsize = (20, 12), title = company_name + " - SMA{} | SMA{}".format(sma_s, sma_l), fontsize = 14)
plt.vlines(x = df.loc[position_sma_buy].loc["03-2019":].index, ymin = df.loc["03-2019":].close.min(), ymax = df.loc["03-2019":].close.max(), color = "C9", label = "매수신호")
plt.vlines(x = df.loc[position_sma_sell].loc["03-2019":].index, ymin = df.loc["03-2019":].close.min(), ymax = df.loc["03-2019":].close.max(), color = "C3", label = "매도신호")
plt.legend(fontsize = 14)
plt.show()

# EMA 전략 시각화하기
df.loc["03-2019":, ["close", "EMA_S", "EMA_L"]].plot(figsize = (20, 12), title = company_name + " - EMA{} | EMA{}".format(ema_s, ema_l), fontsize = 14)
plt.vlines(x = df.loc[position_ema_buy].loc["03-2019":].index, ymin = df.loc["03-2019":].close.min(), ymax = df.loc["03-2019":].close.max(), color = "C9", label = "매수신호")
plt.vlines(x = df.loc[position_ema_sell].loc["03-2019":].index, ymin = df.loc["03-2019":].close.min(), ymax = df.loc["03-2019":].close.max(), color = "C3", label = "매도신호")
plt.legend(fontsize = 14)
plt.show()

# SMA-EMA 전략 시각화하기
df.loc["03-2019":, ["close", "SMA_S", "EMA_S"]].plot(figsize = (20, 12), title = company_name + " - SMA{} | EMA{}".format(ema_s, ema_l), fontsize = 14)
plt.vlines(x = df.loc[position_smaema_buy].loc["03-2019":].index, ymin = df.loc["03-2019":].close.min(), ymax = df.loc["03-2019":].close.max(), color = "C9", label = "매수신호")
plt.vlines(x = df.loc[position_smaema_sell].loc["03-2019":].index, ymin = df.loc["03-2019":].close.min(), ymax = df.loc["03-2019":].close.max(), color = "C3", label = "매도신호")
plt.legend(fontsize = 14)
plt.show()

# 로그수익률 구하기
df["returns"] = np.log(df.close.div(df.close.shift(1))) # 종가(close)기준 전날 대비 로그수익률(매수 후 보유 전략)
df.dropna(inplace = True)
df["creturns"] = df.returns.cumsum().apply(np.exp) # 종가 기준 전날 대비 누적 로그수익률
