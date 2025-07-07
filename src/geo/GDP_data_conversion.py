"""
把NODASS下載之GDP浮標資料轉為GeoJSON和KML格式
點位標示時間、地點、SST
線段標示開始時間、平均SST
"""
import csv, json
import xml.etree.ElementTree as ET

# === 路徑檔名設定 ===
csv_path = '202507021522_export_GDP_22943_result.csv'
geojson_path = 'output.json'
kml_path = 'output.kml'

# === 前置準備 ===
features = []
line_coords = []
total_sst = 0.0
count = 0

# === 讀取並整理CSV資料 ===
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            # 整理資料為固定格式
            lon = float(row['CenterLongitude'])
            lat = float(row['CenterLatitude'])
            time = row['time'].strip().replace('/', '-').replace(' ', 'T')  # ISO 8601
            sst = float(row['sst'])
            
            # 加入點資料
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"time": time, "sst": sst}
            })
            
            # 準備線段座標
            line_coords.append([lon, lat])
            
            # 計算sst平均前置作業
            total_sst += sst
            count += 1
        except (ValueError, KeyError):
            continue

# 計算平均sst
mean_sst = total_sst / count

# 把線段加入features
feature_line = {
    "type": "Feature",
    "geometry": {
        "type": "LineString",
        "coordinates": line_coords
    },
    "properties": {
        "time_first": features[0]['properties']['time'],
        "time_last": features[-1]['properties']['time'],
        "sst_avg": round(mean_sst * 10) / 10
    }
}
features.append(feature_line)

# 建立整個GeoJSON物件
geojson = {
    "type": "FeatureCollection",
    "features": features
}

# 輸出GeoJSON
with open(geojson_path, 'w', encoding='utf-8') as f:
    json.dump(geojson, f, indent=2)

# === 建立KML ===
kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
document = ET.SubElement(kml, "Document")

for feature in features:
    # 建立圖形容器
    placemark = ET.SubElement(document, "Placemark")
    geometry = feature["geometry"]
    props = feature["properties"]
    
    # 建立KML點位
    if geometry["type"] == "Point":
        ET.SubElement(placemark, "name").text = props["time"]
        ET.SubElement(placemark, "description").text = f"SST: {props['sst']}"

        # 建立點位
        point = ET.SubElement(placemark, "Point")
        lon, lat = geometry["coordinates"]
        ET.SubElement(point, "coordinates").text = f"{lon},{lat}"
    
    # 建立KML線段
    elif geometry["type"] == "LineString":
        ET.SubElement(placemark, "name").text = "線段"
        ET.SubElement(placemark, "description").text = (
            f"Start: {props['time_first']}<br/>"
            f"End: {props['time_last']}<br/>"
            f"Mean SST: {props['sst_avg']}"
        )
        # 建立線段
        linestring = ET.SubElement(placemark, "LineString")
        ET.SubElement(linestring, "tessellate").text = "1"
        coords_text = ' '.join(f"{lon},{lat}" for lon, lat in geometry["coordinates"])
        ET.SubElement(linestring, "coordinates").text = coords_text

# 輸出KML
ET.ElementTree(kml).write(kml_path, encoding='utf-8', xml_declaration=True)
