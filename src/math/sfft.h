#ifndef SFFT_H
#define SFFT_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
  double real;
  double imaginary;
} Complex;

void fft(Complex* complexes, size_t size);
void hann_window(double* window, size_t size);
double* averaged_spectrum(const double* samples, size_t size, size_t* output_size, size_t fft_size, size_t hop_size, size_t max_frames);
double base_frequency(const double* samples, size_t size, size_t sample_rate);

#endif
