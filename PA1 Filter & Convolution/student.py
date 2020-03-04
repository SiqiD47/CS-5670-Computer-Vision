# Siqi Dai sd854
# Junyang Ma jm2672

import math
import numpy as np
import PIL
from matplotlib import pyplot as plt
from PIL import Image


def read_image(image_path):
  """Read an image into a numpy array.

  Args:
    image_path: Path to the image file.

  Returns:
    Numpy array containing the image
  """
  im = Image.open(image_path).convert("RGB")
  im = np.uint8(np.array(im))
  return im


def write_image(image, out_path):
  """Writes a numpy array as an image file.
  
  Args:
    image: Numpy array containing image to write
    out_path: Path for the output image
  """
  im = Image.fromarray(image)
  im.save(out_path)


def display_image(image):
  """Displays a grayscale image using matplotlib.

  Args:
    image: HxW Numpy array containing image to display.
  """
  plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
  plt.show()


def convert_to_grayscale(image):
  """Convert an RGB image to grayscale.

  Args:
    image: HxWx3 uint8-type Numpy array containing the RGB image to convert.

  Returns:
    uint8-type Numpy array containing the image in grayscale
  """
  H, W, _ = image.shape
  res = np.zeros((H, W), dtype=np.uint8)
  for i in range(H):
    for j in range(W):
      R, G, B = image[i][j]
      res[i][j] = R * 299/1000 + G * 587/1000 + B * 114/1000
  return res


def convert_to_float(image):
  """Convert an image from 8-bit integer to 64-bit float format

  Args:
    image: Integer-valued numpy array with values in [0, 255]
  Returns:
    Float-valued numpy array with values in [0, 1]
  """
  res = image.astype(np.float64)
  return res / 255


def convolution(image, kernel):
  """Convolves image with kernel.

  The image should be zero-padded so that the input and output image sizes
  are equal.
  Args:
    image: HxW Numpy array, the grayscale image to convolve
    kernel: hxw numpy array
  Returns:
    image after performing convolution
  """
  kernel = np.flip(kernel)
  H, W = image.shape
  h, w = kernel.shape
  image_padded = np.zeros((H+h-1, W+w-1))
  image_padded[(h-1)//2:-(h-1)//2, (w-1)//2:-(w-1)//2] = image  # zero-padded image
  res = np.zeros((H, W), dtype=np.float64)
  for i in range(H):
    for j in range(W):
      res[i][j] = np.sum(kernel * image_padded[i:i+h, j:j+w])
  return res


# helper function to generate a Gaussian kernel
def gaussian_kernel(ksize=3, sigma=1.0):
  kernel = np.zeros((ksize, ksize), dtype=np.float64)
  for i in range(-(ksize//2), ksize//2+1):
    for j in range(-(ksize//2), ksize//2+1):
      kernel[i+ksize//2][j+ksize//2] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
  return kernel / np.sum(kernel)  # normalization


def gaussian_blur(image, ksize=3, sigma=1.0):
  """Blurs image by convolving it with a gaussian kernel.

  Args:
    image: HxW Numpy array, the grayscale image to blur
    ksize: size of the gaussian kernel
    sigma: variance for generating the gaussian kernel

  Returns:
    The blurred image
  """
  kernel = gaussian_kernel(ksize, sigma)
  return convolution(image, kernel)


def sobel_filter(image):
  """Detects image edges using the sobel filter.

  The sobel filter uses two kernels to compute the vertical and horizontal
  gradients of the image. The two kernels are:
  G_x = [-1 0 1]      G_y = [-1 -2 -1]
        [-2 0 2]            [ 0  0  0]
        [-1 0 1]            [ 1  2  1]
  
  After computing the two gradients, the image edges can be obtained by
  computing the gradient magnitude.

  Args:
    image: HxW Numpy array, the grayscale image
  Returns:
    HxW Numpy array from applying the sobel filter to image
  """
  G_x = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
  G_y = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
  grad_x = convolution(image, G_x)
  grad_y = convolution(image, G_y)
  H, W = image.shape
  res = np.zeros((H, W))
  for i in range(H):
    for j in range(W):
      res[i][j] = math.sqrt(grad_x[i][j]**2 + grad_y[i][j]**2)
  return res
 

def dog(image, ksize1=5, sigma1=1.0, ksize2=9, sigma2=2.0):
  """Detects image edges using the difference of gaussians algorithm

  Args:
    image: HxW Numpy array, the grayscale image
    ksize1: size of the first gaussian kernel
    sigma1: variance of the first gaussian kernel
    ksize2: size of the second gaussian kernel
    sigma2: variance of the second gaussian kernel
  Returns:
    HxW Numpy array from applying difference of gaussians to image
  """
  img1 = gaussian_blur(image, ksize1, sigma1)
  img2 = gaussian_blur(image, ksize2, sigma2)
  return img1 - img2


# helper function to visualize the fourier image
def visualize_fourier(fimg):
  H, W = fimg.shape
  res = np.zeros((H, W))
  for i in range(H):
    for j in range(W):
      res[i][j] = math.log(abs(fimg[i][j])) * 255
  display_image(res)


def dft(image):
  """Computes the discrete fourier transform of image

  This function should return the same result as
  np.fft.fftshift(np.fft.fft2(image)). You may assume that
  image dimensions will always be even.

  Args:
    image: HxW Numpy array, the grayscale image
  Returns:
    NxW complex Numpy array, the fourier transform of the image
  """
  H, W = image.shape
  F = np.zeros((H, W), dtype=np.complex128)
  mid_i = H // 2
  mid_j = W // 2
  for i in range(H):
    for j in range(W):
      k = i - mid_i
      l = j - mid_j
      for x in range(H):
        for y in range(W):
          F[i][j] += image[x][y] * np.exp(- 1j * 2 * math.pi * (float(k * x) / H + float(l * y) / W))
  return F


def idft(ft_image):
  """Computes the inverse discrete fourier transform of ft_image.

  For this assignment, the complex component of the output should be ignored.
  The returned array should NOT be complex. The real component should be
  the same result as np.fft.ifft2(np.fft.ifftshift(ft_image)). You
  may assume that image dimensions will always be even.

  Args:
    ft_image: HxW complex Numpy array, a fourier image
  Returns:
    NxW float Numpy array, the inverse fourier transform
  """
  H, W = ft_image.shape
  f = np.zeros((H, W), dtype=np.float64)
  mid_i = H // 2
  mid_j = W // 2
  for x in range(H):
    for y in range(W):
      for i in range(H):
        for j in range(W):
          k = i - mid_i
          l = j - mid_j
          f[x][y] += ft_image[k][l] * np.exp(1j * 2 * math.pi * (float(k * x) / H + float(l * y) / W))
  return abs(f) / (W * H)


def visualize_kernels():
  """Visualizes your implemented kernels.

  This function should read example.png, convert it to grayscale and float-type,
  and run the functions gaussian_blur, sobel_filter, and dog over it. For each function,
  visualize the result and save it as example_{function_name}.png e.g. example_dog.png.
  This function does not need to return anything.
  """
  img_orig = read_image('example.png')
  img_grayscale = convert_to_grayscale(img_orig)
  img = convert_to_float(img_grayscale)

  img_gaussian = gaussian_blur(img)
  img_gaussian = ((img_gaussian - img_gaussian.min()) * (1 / (img_gaussian.max() - img_gaussian.min())) * 255).astype('uint8')  # scale to 0-255
  display_image(img_gaussian)
  write_image(img_gaussian, 'example_gaussian_blur.png')

  img_sobel = sobel_filter(img)
  img_sobel = ((img_sobel - img_sobel.min()) * (1 / (img_sobel.max() - img_sobel.min())) * 255).astype('uint8')  # scale to 0-255
  display_image(img_sobel)
  write_image(img_sobel, 'example_sobel_filter.png')

  img_dog = dog(img)
  img_dog = ((img_dog - img_dog.min()) * (1 / (img_dog.max() - img_dog.min())) * 255).astype('uint8')  # scale to 0-255
  display_image(img_dog)
  write_image(img_dog, 'example_dog.png')


def visualize_dft():
  """Visualizes the discrete fourier transform.

  This function should read example.png, convert it to grayscale and float-type,
  and run dft on it. Try masking out parts of the fourier transform image and
  recovering the original image using idft. Can you create a blurry version
  of the original image? Visualize the blurry image and save it as example_blurry.png.
  This function does not need to return anything.
  """
  ksize = 3
  img_orig = read_image('example_small.png')
  img_grayscale = convert_to_grayscale(img_orig)
  img = convert_to_float(img_grayscale)
  H, W = img.shape[0], img.shape[1]
  img = np.pad(img, ((0, ksize - 1), (0, ksize - 1)), 'constant')
  res_dft = dft(img)

  gaussian_filter = gaussian_kernel(ksize=ksize, sigma=3)
  gaussian_filter = np.pad(gaussian_filter, ((0, H - 1), (0, W - 1)), 'constant')
  gaussian_filter_dft = dft(gaussian_filter)

  res_dft_masked = np.multiply(res_dft, gaussian_filter_dft)
  ans = idft(res_dft_masked)
  ans = ((ans - ans.min()) * (1 / (ans.max() - ans.min())) * 255).astype('uint8')  # scale to 0-255
  ans = ans[(ksize - 1) // 2:-(ksize - 1) // 2, (ksize - 1) // 2:-(ksize - 1) // 2]
  display_image(ans)
  write_image(ans, 'example_blurry.png')
