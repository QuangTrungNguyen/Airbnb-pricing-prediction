#!/usr/bin/python3

import sys
import csv
import pandas as pd
import json
import requests

# static configuration:
url = 'https://mappify.io/api/rpc/coordinates/reversegeocode/'
apiKey = 'XXXXXXXX'  # the apiKey is hidden here
headers = {
    'Content-Type': 'application/json'
}


def get_geodata(lat, lon):
    coordinate = {}
    coordinate['lat'] = lat
    coordinate['lon'] = lon
    coordinate['apiKey'] = apiKey
    json_data = json.dumps(coordinate)
    response = requests.post(url, data=json_data, headers=headers)

    return response.json()


def extract_location_details():
    df = pd.read_csv('test.csv')

    locations = []
    for index, row in df.iterrows():
        loc = {}
        id, lat, lon = row['Id'], row['latitude'], row['longitude']
        detail = get_geodata(lat, lon)

        loc['id'] = id
        loc['lat'] = lat
        loc['lon'] = lon
        loc['suburb'] = (detail['result']['suburb']).strip()
        loc['postcode'] = (detail['result']['postCode']).strip()
        locations.append(loc)

    df = pd.read_csv('train.csv')

    for index, row in df.iterrows():
        loc = {}
        id, lat, lon = row['Id'], row['latitude'], row['longitude']
        detail = get_geodata(lat, lon)

        loc['id'] = id
        loc['lat'] = lat
        loc['lon'] = lon
        loc['suburb'] = (detail['result']['suburb']).strip()
        loc['postcode'] = (detail['result']['postCode']).strip()
        locations.append(loc)


    with open('extracted_location.csv', 'w') as csvfile:
        fieldnames = ['id', 'lat', 'lon', 'detail', 'suburb', 'postcode']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for loc in locations:
            writer.writerow(loc)


if __name__ == "__main__":
    extract_location_details()
