# JPEG-Like Image Encoder/Decoder Project
### Overview
This project is designed to develop a JPEG-like image encoder and decoder that employs lossy image compression techniques to significantly reduce the size of image files. The objective is to understand and implement the underlying processes of the JPEG compression algorithm, focusing on the reduction of data that is less perceptible to the human eye, thereby achieving efficient compression with minimal perceived loss of image quality.

### Project Structure

Loading and Saving Images: Implement functions to input and output image files.

Color Space Conversion: Convert images from the RGB color space to the YCbCr color space, which separates luminance (Y) from chrominance (Cb and Cr).

Chrominance Subsampling: Implement subsampling for chrominance channels to reduce the image data based on the human eye’s lower sensitivity to color detail compared to luminance.

DCT Implementation: Apply a 2D DCT to each 8x8 block of the image. This transforms the image from the spatial domain to the frequency domain, representing the image as a sum of cosine functions with varying frequencies and amplitudes.

Quantization: Quantize the DCT coefficients using a quantization matrix to prioritize lower frequencies and attenuate higher frequencies, which are less perceptible to the human eye.

Run-Length Encoding (RLE) and Arithmetic Coding: Implement these techniques to efficiently encode the quantized data, significantly reducing the size without losing critical information.

