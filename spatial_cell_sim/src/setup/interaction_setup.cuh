#pragma once

#include <random>

#include <cuda_runtime.h>
#include <crt/math_functions.h>

#include "../types.cuh"
#include "../constants.cuh"
#include "../math.cuh"


template <typename T> void
linkParticlesSerially(
	int start,
	int end,
	int interactionType,
	T* h_Particles,
	std::unordered_map<int, ParticleInteractionInfo> *complexificationInfo,
	const Config* config,
	std::function<double()> rng
) {
	/*int interactionGroup = -1;
	if (interactionType > 10) {
		ParticleInteractionInfo pii = (*complexificationInfo)[interactionType];
		interactionGroup = pii.group;
	}*/
	for (int i = start + 1; i < end; i++) {
		T* p = &h_Particles[i];
		T* interactionPartner = &h_Particles[i - 1];

		// Add interaction for the partner
		interactionPartner->interactions[interactionPartner->nActiveInteractions].type = interactionType;
		interactionPartner->interactions[interactionPartner->nActiveInteractions].group = INTERACTION_GROUP_FORWARD;
		interactionPartner->interactions[interactionPartner->nActiveInteractions].partnerId = p->id;
		interactionPartner->nActiveInteractions++;
		// Add interaction for the current particle
		p->interactions[p->nActiveInteractions].type = interactionType;
		p->interactions[p->nActiveInteractions].group = INTERACTION_GROUP_BACKWARD;
		p->interactions[p->nActiveInteractions].partnerId = interactionPartner->id;
		p->nActiveInteractions++;
	}
}
