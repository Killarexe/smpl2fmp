#include "sfft.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void fft(Complex* complexes, size_t size) {
  for (size_t i = 1, j = 0; i < size; i++) {
    size_t bit = size >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;

    if (i < j) {
      Complex temp = complexes[i];
      complexes[i] = complexes[j];
      complexes[j] = temp;
    }
  }

  for (size_t length = 2; length <= size; length <<= 1) {
    double angle = -2.0 * M_PI / length;
    Complex wlength = { cos(angle), sin(angle) };
    for (size_t i = 0; i < size; i += length) {
      Complex w = { 1.0, 0.0 };
      for (size_t j = 0; j < length / 2; j++) {
        Complex u = complexes[i + j];
        Complex v = {
          complexes[i + j + length / 2].real * w.real - complexes[i + j + length / 2].imaginary * w.imaginary,
          complexes[i + j + length / 2].real * w.imaginary + complexes[i + j + length / 2].imaginary * w.real
        };

        complexes[i + j] = (Complex){ u.real + v.real, u.imaginary + v.imaginary };
        complexes[i + j + length / 2] = (Complex){ u.real - v.real, u.imaginary - v.imaginary };

        double wreal = w.real * wlength.real - w.imaginary * wlength.imaginary;
        double wimagenary = w.real * wlength.imaginary + w.imaginary * wlength.real; 

        w.real = wreal;
        w.imaginary = wimagenary;
      }
    }
  }
}

void hann_window(double* window, size_t size) {
  for (size_t i = 0; i < size; i++) {
    window[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (size - 1)));
  }
}

double* averaged_spectrum(const double* samples, size_t size, size_t* output_size, size_t fft_size, size_t hop_size, size_t max_frames) {
  double* window = malloc(fft_size * sizeof(double));
  if (!window) {
    return NULL;
  }
  hann_window(window, fft_size);
  
  size_t average_size = fft_size / 2 + 1;
  double* average = calloc(average_size, sizeof(double));
  if (!average) {
    free(window);
    return NULL;
  }

  Complex* buffer = malloc(fft_size * sizeof(Complex));
  if (!buffer) {
    free(window);
    free(average);
    return NULL;
  }
  
  size_t frames = 0;
  for (size_t start = 0; start + fft_size <= size && frames < max_frames; start += hop_size, frames++) {
    for (size_t i = 0; i < fft_size; i++) {
      buffer[i].real = samples[start + i] * window[i];
      buffer[i].imaginary = 0.0f;
    }

    fft(buffer, fft_size);

    for (size_t i = 0; i < average_size; i++) {
      double mangnitude = sqrt(buffer[i].real * buffer[i].real + buffer[i].imaginary * buffer[i].imaginary);
      average[i] += mangnitude; 
    }
  }

  if (frames > 0) {
    for (size_t i = 0; i < average_size; i++) {
      average[i] /= frames;
    }
  }

  for (size_t i = 0; i < average_size; i++) {
    average[i] = 20.0 * log10(average[i] + 1e-9);
  }

  free(buffer);
  free(window);
  *output_size = frames;
  return average;
}

double base_frequency(const double* samples, size_t size, size_t sample_rate) {
  size_t fft_size = pow(2, ceil(log((double)size) / log(2)));
  Complex* buffer = malloc(fft_size * sizeof(Complex));
  
  for (size_t i = 0; i < fft_size; i++) {
    buffer[i].real = i < size ? samples[i] : 0.0;
    buffer[i].imaginary = 0.0;
  }

  fft(buffer, fft_size);

  double max_magnitude = 0.0;
  size_t base_frequency_index = 0;
  for (size_t i = 1; i < fft_size / 2 + 1; i++) {
    Complex complex = buffer[i];
    double magnitude = sqrt(complex.real * complex.real + complex.imaginary * complex.imaginary);
    if (magnitude > max_magnitude) {
      base_frequency_index = i;
      max_magnitude = magnitude;
    }
  }

  free(buffer);

  return (double)base_frequency_index * (double)sample_rate / (double)fft_size;
}
