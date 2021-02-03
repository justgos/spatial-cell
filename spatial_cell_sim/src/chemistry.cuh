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
