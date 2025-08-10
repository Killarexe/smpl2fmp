#pragma once

#include <fftw3.h>
#include <vector>

class FFTProcessor {
  private:
    size_t sampleSize;
    size_t numSamples;
    size_t outputSize;
    
    double* inputBuffer;
    fftw_complex* outputBuffer;
    fftw_plan plan;

  public:
    FFTProcessor(size_t sampleSize, size_t numSamples);
    ~FFTProcessor();
    
    double* getInput(size_t index);
    void setSample(size_t index, const std::vector<double>& sample);
    void execute();
    std::vector<double> getMagnitude(size_t index);
};
