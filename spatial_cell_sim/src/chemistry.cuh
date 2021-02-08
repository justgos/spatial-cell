#pragma once

#include <fstream>

#include <json/json.h>

#include "./types.cuh"
    

std::unordered_map<int, ParticleTypeInfo>*
loadParticleTypeInfo() {
    auto particleTypeInfo = new std::unordered_map<int, ParticleTypeInfo>();

    std::ifstream largeMoleculeFile("../universe-config/gen/large-molecules.json");
    Json::Value largeMoleculeJson;
    largeMoleculeFile >> largeMoleculeJson;

    for (int i = 0; i < largeMoleculeJson.size(); i++) {
        auto m = largeMoleculeJson[i];
        int type = m["type"].asInt();

        particleTypeInfo->insert(std::pair<int, ParticleTypeInfo>(type, ParticleTypeInfo(
            m["category"].asCString(),
            m["name"].asCString(),
            m["radius"].asFloat(),
            m.get("hydrophobic", false).asInt()
        )));
    }

    return particleTypeInfo;
}

/*
* Transform the particle type map into a flat array with index<->id identity
*/
SingleBuffer<MinimalParticleTypeInfo>*
flattenParticleTypeInfo(std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo) {
    int maxIndex = 0;
    for (auto it = particleTypeInfo->begin(); it != particleTypeInfo->end(); it++) {
        maxIndex = max(it->first, maxIndex);
    }

    auto flattenedInfo = new SingleBuffer<MinimalParticleTypeInfo>(maxIndex + 1);

    for (auto it = particleTypeInfo->begin(); it != particleTypeInfo->end(); it++) {
        flattenedInfo->h_Current[it->first].radius = it->second.radius;
        flattenedInfo->h_Current[it->first].hydrophobic = it->second.hydrophobic;
    }

    return flattenedInfo;
}
