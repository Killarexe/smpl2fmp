#include "diffs.h"

#include <math.h>

double cosine_similarity(const double* a, const double* b, size_t size) {
  double dot_product = 0.0;
  double total_a = 0.0;
  double total_b = 0.0;

  for (size_t i = 0; i < size; i++) {
    dot_product += a[i] * b[i];
    total_a += a[i] * a[i];
    total_b += b[i] * b[i];
  }

  if (total_a == 0.0 || total_b == 0.0) {
    return 0.0;
  }

  return dot_product / (sqrt(total_a) * sqrt(total_b));
}

double spectral_distance(const double* a, const double* b, size_t size) {
  double sum_distance = 0.0;
  double sum_normal = 0.0;

  for (size_t i = 0; i < size; i++) {
    double distance = a[i] - b[i];
    sum_distance += distance * distance;
    sum_normal += a[i] * a[i] + b[i] * b[i];
  }

  if (sum_normal == 0.0) {
    return 1.0;
  }

  return 1.0 - sqrt(sum_distance / sum_normal);
}

double spectral_flux(const double* a, const double* b, size_t size) {
  double flux = 0.0;
  double total = 0.0;

  for (size_t i = 0; i < size; i++) {
    flux += fabs(a[i] - b[i]);
    total += a[i] + b[i];
  }

  if (total == 0.0) {
    return 1.0;
  }

  return 1.0 - (flux / total);
}

double harmonic_peak_score(const double* target, const double* synth, size_t size, double base_frequency, double sample_rate, size_t fft_size) {
  double score = 0.0;
  size_t count = 0;
  double bin_width = sample_rate / (double)fft_size;
  size_t window = 2;

  for (size_t h = 0; h * base_frequency < (sample_rate / 2.0); h++) {
    size_t bin = (size_t)round(h * base_frequency / bin_width);
    if (bin >= size) {
      break;
    }

    size_t low = bin > window ? bin - window : 0;
    size_t high = bin + window < size ? bin + window : size - 1;

    double target_peak = 0.0;
    double synth_peak = 0.0;
    for (size_t i = low; i <= high; i++) {
      if (target[i] > target_peak) {
        target_peak = target[i];
      }
      if (synth[i] > synth_peak) {
        synth_peak = synth[i];
      }
    }

    if (target_peak > 1e-9) {
      double ratio = fmin(synth_peak / target_peak, target_peak / synth_peak);
      score += ratio;
      count++;
    }
  }

  return count > 0 ? score / count : 0.0;
}
