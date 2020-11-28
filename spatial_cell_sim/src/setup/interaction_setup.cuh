#pragma once

#include <random>

#include <cuda_runtime.h>
#include <crt/math_functions.h>

#include "../types.cuh"
#include "../constants.cuh"
#include "../math.cuh"


void
linkParticlesSerially(
	int start,
	int end,
	Particle* h_Particles,
	const Config* config,
	std::function<double()> rng
) {
	for (int i = start + 1; i < end; i++) {
		Particle* p = &h_Particles[i];
		Particle* interactionPartner = &h_Particles[i - 1];

		// Add interaction for the partner
		interactionPartner->interactions[interactionPartner->nActiveInteractions].type = 0;
		interactionPartner->interactions[interactionPartner->nActiveInteractions].partnerId = i;
		interactionPartner->nActiveInteractions++;
		// Add interaction for the current particle
		p->interactions[p->nActiveInteractions].type = 0;
		p->interactions[p->nActiveInteractions].partnerId = interactionPartner->id;
		p->nActiveInteractions++;
	}
}
