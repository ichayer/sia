import requests
import math
from colr import color

# LEER IMPORTANTE

# Obviamente hay toda una ciencia atrás de los colores y parece ser que hay "Color Spaces" (como RGB, XYZ y LAB) cada uno con sus características. El espacio LAB sirve bastante para combinar colores por lo que uso una página para convertir de RGB a LAB. 

# Usando la página va rápido pero desde mi compu con requests tarda bastante, por lo que en un algoritmo genético podría ser un problema de tiempo. Si no se usa esta página hay que pasar de RGB a XYZ a mano y después de XYZ a LAB a mano también. Se usan matrices y no se que mierda, muchos cálculos que me dieron paja entender. Quizás es lo de requests que tarda tanto, porque en la página no tarda mucho, no sé.

# En el main hay un ejemplo

# Página con ecuaciones de los pasajes entre espacios
# http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html 

# Página para mezclar colores para referencia
# https://pinetools.com/es/mezclar-colores 


class RGBColor:
    def __init__(self, r: int, g: int, b: int) -> None:
        if r<0 or r>255 or g<0 or g>255 or b<0 or b>255:
            raise ValueError('Invalid RGB parameters')
        self.r = r
        self.g = g
        self.b = b
        
    def show(self):
        print(color('  ', fore=(0, 0, 0), back=(self.r, self.g, self.b)))
        

        
class LABColor:
    def __init__(self, l: int, a: int, b: int) -> None:
        if l<0 or l>100 or a<-128 or a>128 or b<-128 or b>128:
            raise ValueError('Invalid RGB parameters')
        self.l = l
        self.a = a
        self.b = b
        

def similitude(c1: RGBColor, c2: RGBColor) -> int:
    l1 = RGBToLAB(c1)
    l2 = RGBToLAB(c2)
    return math.sqrt((l1.l-l2.l)*(l1.l-l2.l)+(l1.a-l2.a)*(l1.a-l2.a)+(l1.b-l2.b)*(l1.b-l2.b))


def RGBToLAB(c: RGBColor) -> LABColor:
    rgb = {'R': c.r, 'G': c.g, 'B': c.b}
    url = 'http://colormine.org/api/Rgb/Translate'
    res = requests.post(url, json = rgb)
    l = res.json()["Lab"]["Members"][0]["Value"]
    a = res.json()["Lab"]["Members"][1]["Value"]
    b = res.json()["Lab"]["Members"][2]["Value"]
    return LABColor(l,a,b)

def LABToRGB(c: LABColor) -> RGBColor:
    lab = {'L': c.l, 'A': c.a, 'B': c.b}
    url = 'http://colormine.org/api/Lab/Translate'
    res = requests.post(url, json = lab)
    r = res.json()["Rgb"]["Members"][0]["Value"]
    g = res.json()["Rgb"]["Members"][1]["Value"]
    b = res.json()["Rgb"]["Members"][2]["Value"]
    return RGBColor(r,g,b)

def LABMix(c1: LABColor, c2: LABColor) -> LABColor:
    return LABColor((c1.l+c2.l)/2, (c1.a+c2.a)/2, (c1.b+c2.b)/2)