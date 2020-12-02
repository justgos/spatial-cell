#pragma once

#include <fstream>

#include "./types.cuh"
#include "./memory.cuh"


class FileStorage {
private:
	std::string fileName;
	std::ofstream fout;
	Config *config;

public:
	FileStorage(std::string fileName, Config* config)
		: fileName(fileName), config(config)
	{
		fout.open(fileName, std::ios::binary | std::ios::out);
	}

	void
	writeHeader() {
		fout.write((char*)&config->simSize, sizeof(float));
		fout.write((char*)&config->numParticles, sizeof(unsigned int));
		fout.write((char*)&config->numMetabolicParticles, sizeof(unsigned int));
	}

	template <typename ParticleType, typename MetabolicParticleType>
	void
	writeFrame(
		const SingleBuffer<ParticleType> *particles,
		const SingleBuffer<MetabolicParticleType> *metabolicParticles
	) {
		fout.write((char*)&config->numParticles, sizeof(unsigned int));
		fout.write((char*)particles->h_Current, particles->size);
		fout.write((char*)&config->numMetabolicParticles, sizeof(unsigned int));
		fout.write((char*)metabolicParticles->h_Current, metabolicParticles->size);
	}

	~FileStorage() {
		fout.close();
	}
};
