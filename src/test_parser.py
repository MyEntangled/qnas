import math
import argparse
def cylinder_volume(radius, height):
    vol = (math.pi) * (radius ** 2) * height
    return vol

parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
parser.add_argument('-r', '--radius', type=int, metavar='', required=True, help='Radius of Cylinder')
parser.add_argument('height', type=int, metavar='', help='Height of Cylinder')
#parser.add_argument('-r', 'g')
args = parser.parse_args()

if __name__ == '__main__':
    print(cylinder_volume(args.radius, args.height))
