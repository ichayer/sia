import requests
import math

class RGBColor:
    def __init__(self, r: int, g: int, b: int) -> None:
        if r<0 or r>255 or g<0 or g>255 or b<0 or b>255:
            raise ValueError('Invalid RGB parameters')
        self.r = r
        self.g = g
        self.b = b
        
class LABColor:
    def __init__(self, l: int, a: int, b: int) -> None:
        if l<0 or l>100 or a<-128 or a>128 or b<-128 or b>128:
            raise ValueError('Invalid RGB parameters')
        self.l = l
        self.a = a
        self.b = b
        

def similitude(c1: RGBColor, c2: RGBColor) -> int:
    l1 = RGBToLab(c1)
    l2 = RGBToLab(c2)
    return math.sqrt((l1.l-l2.l)*(l1.l-l2.l)+(l1.a-l2.a)*(l1.a-l2.a)+(l1.b-l2.b)*(l1.b-l2.b))


def RGBToLab(c: RGBColor) -> LABColor:
    url = 'http://colormine.org/api/Rgb/Translate'
    rgb = {'R': c.r, 'G': c.g, 'B': c.b}
    res = requests.post(url, json = rgb)
    l = res.json()["Lab"]["Members"][0]["Value"]
    a = res.json()["Lab"]["Members"][1]["Value"]
    b = res.json()["Lab"]["Members"][2]["Value"]
    return LABColor(l,a,b)