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
	}

	void
	writeFrame(
		const DoubleBuffer<Particle> *particles
	) {
		fout.write((char*)&config->numParticles, sizeof(unsigned int));
		fout.write((char*)particles->h_Current, particles->size);
	}

	~FileStorage() {
		fout.close();
	}
};