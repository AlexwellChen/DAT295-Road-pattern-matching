
import os
import random
import argparse
import math
from PIL import Image
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from io import BytesIO
from tqdm import tqdm
import requests


class Solution:
    def __init__(self):
        self.proxy = None
        self.proxy_list = []
        self.build_proxy_list()
        self.cnt = 0

    def coordinate_to_tile(self, lat, lon, zoom):
        lat_rad = lat * math.pi / 180.0
        n = math.pow(2, zoom)
        x_tile = n * ((lon + 180) / 360.0)
        y_tile = n * (1-(math.log(math.tan(lat_rad) + 1 /
                                  math.cos(lat_rad)) / math.pi)) / 2.0

        return round(x_tile), round(y_tile)

    def scraper(self,  time, zoom_level, region, filename):
        output_file = os.path.dirname(
            os.path.realpath(__file__)) + "/" + filename
        w = region['x_end'] - region['x_start']
        h = region['y_end'] - region['y_start']

        url_template = (
            "http://mt%i.google.com/vt?lyrs=h,traffic|seconds_into_week:%i&x=%i&y=%i&z=%i&apistyle=p.v%%3Aoff")
        image_out = Image.new('RGB', (256*w, 256*h), color=(255, 255, 255))

        # proxy = self.get_proxy()

        with tqdm(total=w * h, desc="SINGLE TILE", leave=False) as pbar:
            for i in range(w):
                for j in range(h):
                    while True:
                        x = region['x_start'] + i
                        y = region['y_start'] + j
                        url = url_template % (
                            random.randrange(4), time, x, y, zoom_level)
                        print(url)
                        proxy_host = self.proxy['ip'] + \
                            ':' + self.proxy['port']

                        # print(url)

                        try:
                            im = self.get_traffic_image(url, proxy_host)
                            break
                        except Exception as err:
                            self.proxy = self.get_proxy()
                            # print(proxy)
                            continue
                    im = im.convert('RGBA')
                    image_out.paste(im, (256*i, 256*j), im)
                    pbar.update(1)
                image_out.save(output_file)

    def build_proxy_list(self):
        self.proxy_list = []
        # url = "https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=lastChecked&sort_type=desc&google=true"
        # # # url = 'https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=responseTime&sort_type=asc&protocols=http'

        # proxies_req = requests.get(url)
        # for d in proxies_req.json()['data']:
        #     # print(d['ip'], d['port'])
        #     self.proxy_list.append({
        #         'ip': d['ip'],
        #         'port': d['port']
        #     })

        ua = UserAgent()
        # proxies_req = Request('https://www.sslproxies.org/')
        proxies_req = Request('https://www.google-proxy.net/')
        proxies_req.add_header('User-Agent', ua.random)
        proxies_doc = urlopen(proxies_req).read().decode('utf8')
        soup = BeautifulSoup(proxies_doc, 'html.parser')
        proxies_table = soup.find(class_='fpl-list')
        # proxies_table = soup.find(class_='table')

        for row in proxies_table.tbody.find_all('tr'):
            self.proxy_list.append({
                'ip':   row.find_all('td')[0].string,
                'port': row.find_all('td')[1].string
            })
        # print(len(self.proxy_list))
        self.proxy = self.proxy_list[0]

    def build_tile_region(self, start_lat, start_lon, end_lat, end_lon, zoom):
        sx, sy = self.coordinate_to_tile(start_lat, start_lon, zoom)
        ex, ey = self.coordinate_to_tile(end_lat, end_lon, zoom)
        if(sx > ex):
            sx, ex = ex, sx
        if(ey > ey):
            sy, ey = ey, sy

        return {
            'x_start': sx,
            'x_end': ex,
            'y_start': sy,
            'y_end': ey
        }

    def get_traffic_image(self, url, proxy):
        request = Request(url)
        request.set_proxy(proxy, 'http')
        image_data = urlopen(request, timeout=3).read()

        return Image.open(BytesIO(image_data))

    def get_proxy(self):
        self.cnt += 1
        if(self.cnt == min(75, len(self.proxy_list))):
            self.cnt = 0
            self.build_proxy_list()
        return self.proxy_list[self.cnt]


cnt = 0

if __name__ == '__main__':
    sol = Solution()

    level = 15
    # time = 64800
    region = sol.build_tile_region(
        42.3258,  -83.674, 42.2203052778, -83.8042902778,   level)

    sol.scraper(6*3600, 15, region, 'test6AM.png')
    # print(region)
    # start_point = 987
    # loop = tqdm(range(start_point, 1008), desc="Iterate ")
    # for time in loop:
    #     loop.set_description("Iteration " + str(time))
    #     filename = "./output/%s_%s_%s_%s_z%s_t%s.png" % (
    #         region['x_start'],
    #         region['x_end'],
    #         region['y_start'],
    #         region['y_end'],
    #         level,
    #         (time)*600,
    #     )
    #     sol.scraper(time, level, region, filename)
