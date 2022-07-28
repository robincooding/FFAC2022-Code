# 라이브러리 호출
import yfinance as yf
import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

yf.pdr_override()   # yahoo finance 활성화
start = dt.datetime(2021,6,1)
now = dt.datetime.now()

stock = input("Enter the stock symbol : ")  # Stock ticker 입력

while stock != "quit":

    df = pdr.get_data_yahoo(stock, start, now)    # 주식 가격 데이터 가져와 데이터프레임에 저장

    df['Low'].plot(label='low')

    pivots = []
    dates = []
    counter = 0
    lastPivot = 0

    rangeInit = max(df['Low'])    # Range 초기화할 최댓값 

    # local minimum 10일 기준으로 계산
    Range = [rangeInit, rangeInit, rangeInit, rangeInit, rangeInit, rangeInit, rangeInit, rangeInit, rangeInit, rangeInit]
    dateRange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in df.index:
        currentMin = min(Range, default=0)
        value = round(df['Low'][i], 2)

        Range = Range[1:9]
        Range.append(value)
        dateRange = dateRange[1:9]
        dateRange.append(i)

        if currentMin == min(Range, default=0):
            counter += 1
        else:
            counter = 0
        if counter == 5:
            lastPivot = currentMin
            dateloc = Range.index(lastPivot)
            lastDate = dateRange[dateloc]

            pivots.append(lastPivot)
            dates.append(lastDate)
        
    print()

    timeD = dt.timedelta(days=30)

    for index in range(len(pivots)):
        print(str(pivots[index]) + ": " + str(dates[index]))
    
        plt.plot_date([dates[index], dates[index] + timeD], [pivots[index], pivots[index]], fmt=",-", linewidth=2)


    plt.show()

    stock = input("Enter the stock symbol : ")