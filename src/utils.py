from LGB import LightGBM
import pvlib 
from meteostat import Point, Hourly 
import pandas as pd

class Feature_Engineering:    
    
    def feature_from_pvlib(lat, lon, start, end, freq):
        """
        特徵建構 : 可看未來 pv資料，以分鐘為單位
        1. solar_position(太陽位置) : zenith-天頂角、azimuth-方位角、elevation-仰角
        2. clear_sky(輻射) : ghi-全球水平輻射、dni-直接法線輻射、dhi-散射水平輻射
        3. airmass(空氣質量) : 光線穿過地球大氣層的長度，可以影響太陽輻射的強度
        4. cloud_cover(雲層覆蓋率)
        """
        try:
            times = pd.date_range(start, end, freq=freq, tz='Asia/Taipei') 
            site = pvlib.location.Location(lat, lon)
            solar_position = site.get_solarposition(times).reset_index() 
            clear_sky = site.get_clearsky(times, model='ineichen').reset_index()
            airmass = pvlib.atmosphere.get_relative_airmass(solar_position['zenith']).reset_index()
            cloud_cover = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon).reset_index()
            final = pd.concat([solar_position, clear_sky, airmass, cloud_cover],axis=1)
            final.columns = ['DateTime', 'apparent_zenith', 'zenith', 'apparent_elevation', 'elevation',
                             'azimuth', 'equation_of_time',
                             'DateTime_2', 'ghi', 'dni', 'dhi',
                             'DateTime_3', 'airmass',
                             'DateTime_4', 'cloud_cover']
            final = final.drop(['DateTime_2','DateTime_3','DateTime_4'], axis=1).reset_index(drop=True)
            return final
        except Exception as e:
            print(f'feature_from_pvlib has error: {e}')
            return None
        
    def feature_from_meteostat(lat, lon, start, end):
        """
        特徵建構 meteostat 包含歷史與未來7日小時氣象資料
        無花蓮 宜蘭 經緯度，以台北代替 (25.09108, 121.5598)
        """
        try:
            location = Point(lat, lon)  
            data = Hourly(location, start, end)
            data = data.fetch()
            data = data[['wspd', 'pres', 'temp', 'rhum']].reset_index()
            data['key'] = data['time'].astype(str).str[:13]
            return data
        except Exception as e:
            print(f'feature_from_meteostat has error: {e}')
            return None
    
    def resample_10T(place, data, feature_name) :
        """ 取10分鐘平均，中位數較差 """
        try:
            use = data.loc[data['地點_x']==place].reset_index(drop=True)
            use['DateTime'] = pd.to_datetime(use['DateTime_x'])
            use.set_index('DateTime', inplace=True)
            use = use[feature_name].resample('10T').mean().reset_index()
            use = use.dropna().reset_index(drop=True)
            use['minutes_min'] = use['DateTime'].astype(str).str[14:16].astype(int)
            use = use.rename(columns={'地點_x': '地點'})
            use['key'] = use['DateTime'].astype(str).str[:16] + '_' + use['地點'].astype(int).astype(str)
            return use
        except Exception as e:
            print(f'resample_10T has error: {e}')
            return None

if __name__ == '__main__':
    out = Feature_Engineering.feature_from_pvlib(lat, lon, start, end)
    out = Feature_Engineering.feature_from_pvlib(lat, lon, start, end)