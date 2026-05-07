#ifndef DIFFS_H
#define DIFFS_H

#include <stddef.h>

double cosine_similarity(const double* a, const double* b, size_t size);

double spectral_distance(const double* a, const double* b, size_t size);

double spectral_flux(const double* a, const double* b, size_t size);

double harmonic_peak_score(const double* target, const double* synth, size_t size, double base_frequency, double sample_rate, size_t fft_size);

#endif
