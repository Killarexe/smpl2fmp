#include "individual.h"
#include "AudioFile/AudioFile.h"
#include <cstdint>

void Individual::saveAudio(std::filesystem::path path, double frequency, double duration, uint32_t sampleRate) {
  AudioFile<double>::AudioBuffer buffer;
  buffer.resize(2);
  buffer[0].resize(sampleRate * duration);
  buffer[1].resize(sampleRate * duration);
  synthetize(frequency, duration, sampleRate, buffer);
  AudioFile<double> audio;
  audio.setSampleRate(sampleRate);
  audio.setAudioBuffer(buffer);
  audio.setAudioBufferSize(buffer.size(), buffer[0].size());
  audio.save(path, AudioFileFormat::Wave);
}
