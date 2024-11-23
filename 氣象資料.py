from meteostat import Point, Hourly # 最多看 未來7天的氣象
from datetime import datetime

# 定義地理位置，這裡以紐約市為例
location = Point(40.7128, -74.0060)  # 緯度和經度

# 設定日期範圍，這裡查詢2024年3月1日的數據
start = datetime(2024, 3, 1)
end = datetime(2024, 3, 1)
start = datetime.strptime(start[:10], '%Y-%m-%d')
end = datetime.strptime(end[:10], '%Y-%m-%d')

# 查詢小時級別的數據
data = Hourly(location, start, end)
data = data.fetch()

# 顯示風速、氣壓、溫度和濕度
print(data[['wspd', 'pres', 'temp', 'rhum']])


