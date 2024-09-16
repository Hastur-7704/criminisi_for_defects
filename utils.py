import numpy


M = numpy.array([[0.4124564, 0.3575761, 0.1804375],
              [0.2126729, 0.7151522, 0.0721750],
              [0.0193339, 0.1191920, 0.9503041]])


def f(im_channel):
    return numpy.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931

def anti_f(im_channel):
    return numpy.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787


def __rgb2xyz__(pixel):
    r, g, b = pixel[0], pixel[1], pixel[2]
    rgb = numpy.array([r, g, b])
    # rgb = rgb / 255.0
    # RGB =numpy.array([gamma(c) for c in rgb])
    XYZ = numpy.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab(pixel):
    xyz = __rgb2xyz__(pixel)
    Lab = numpy.array(__xyz2lab__(xyz))
    return Lab


def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = numpy.array(xyz)
    xyz = xyz * 255
    rgb = numpy.dot(numpy.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = numpy.uint8(numpy.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb


if __name__ == '__main__':
    rgb_original = numpy.array([255, 255, 153])
    lab_transferred = RGB2Lab(rgb_original)
    rgb_restored = Lab2RGB(lab_transferred)
    print("RGB:", type(rgb_original), rgb_original)
    print("Lab:", type(lab_transferred), lab_transferred)
    print("RGB:", type(rgb_restored), rgb_restored)
