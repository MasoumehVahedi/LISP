import numpy as np
import sys


class MortonCode:

    def part1by1(self, n):
        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    def unpart1by1(self, n):
        n &= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0f0f0f0f
        n = (n ^ (n >> 4)) & 0x00ff00ff
        n = (n ^ (n >> 8)) & 0x0000ffff
        return n

    def interleave2(self, y, x):  # Swap the order of x and y
        return self.part1by1(y) | (self.part1by1(x) << 1)

    def deinterleave2(self, n):
        return self.unpart1by1(n), self.unpart1by1(n >> 1)

    def interleave_latlng(self, lat, lng):
        # Latitude = y-coordinate
        # Longitude = x-coordinate
        if lng > 180:
            x = (lng % 180) + 180.0
        elif lng < -180:
            x = (-((-lng) % 180)) + 180.0
        else:
            x = lng + 180.0
        if lat > 90:
            y = (lat % 90) + 90.0
        elif lat < -90:
            y = (-((-lat) % 90)) + 90.0
        else:
            y = lat + 90.0

        x = round(x)
        y = round(y)

        morton_code = self.interleave2(y, x)  #The order of x and y is important
        return morton_code


