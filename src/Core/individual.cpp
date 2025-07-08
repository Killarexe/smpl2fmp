#include "individual.h"
#include "AudioFile/AudioFile.h"
#include <cstdint>

void Individual::saveAudio(std::filesystem::path path, double frequency, double duration, uint32_t sampleRate) {
  AudioFile<double>::AudioBuffer buffer = synthetize(frequency, duration, sampleRate);
  AudioFile<double> audio;
  audio.setSampleRate(sampleRate);
  audio.setAudioBuffer(buffer);
  audio.setAudioBufferSize(buffer.size(), buffer[0].size());
  audio.save(path, AudioFileFormat::Wave);
}
