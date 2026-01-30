//
// Copyright (C) 2025 CM Lee, SJ Ye, Seoul Sational University
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// 	"License"); you may not use this file except in compliance
// 	with the License.You may obtain a copy of the License at
// 
// 	http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.

/**
 * @file    module/transport/buffer_struct.hpp
 * @brief   Enums related to the transport system
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <map>
#include <string>
#include <set>

#include "particles/define_struct.hpp"


namespace mcutil {


	/**
	* @brief kernel launch characteristic (block-wised, warp-wised, or thread-wised)
	*/
	enum class BUFFER_KERNEL_LAUNCH_STRUCTURE {
		VIRTUAL,
		THREAD_WISED,
		WARP_WISED,
		BLOCK_WISED
	};


	enum class BUFFER_KERNEL_PHYSICS_PROPERTY {  // Value is the queue priority (larger is prior)
		SOURCE                 = 1,
		TRANSPORT              = 10,
		CASCADE_INTERACTION    = 1000,
		INTERACTION            = 100000,
		UNUSED                 = 0
	};

    
	enum BUFFER_TYPE {
		// virtual
		SOURCE,
		QMD_SOURCE,      // for the module tester
		// Optix transport
		ELECTRON,
		PHOTON,
		POSITRON,
		GROUP_NEUTRON,   // Groupwised Neutron < 20 MeV
		POINT_NEUTRON,   // Pointwised Neutron < 20 MeV
		POINT_NEUTRON_T, // Pointwised Neutron < 4 eV thermal
		GNEUTRON,        // Generic neutron (cover high)
		GENION,          // Generic ion
		// interactions 
		RELAXATION,
		RAYLEIGH,
		PHOTO,
		COMPTON,
		PAIR,
		EBREM,
		EBREM_SP,
		MOLLER,
		PBREM,
		PBREM_SP,
		BHABHA,
		ANNIHI,
		G_NEU_SECONDARY, // groupwised neutron secondary
		P_NEU_R_1,       // pointwised neutron reaction bucket LAW=1
		P_NEU_R_3,       // pointwised neutron reaction bucket LAW=3
		P_NEU_R_4,       // pointwised neutron reaction bucket LAW=4
		P_NEU_R_5,       // pointwised neutron reaction bucket LAW=5
		P_NEU_R_79,      // pointwised neutron reaction bucket LAW=7 and LAW=9
		P_NEU_R_11,      // pointwised neutron reaction bucket LAW=11
		P_NEU_R_22,      // pointwised neutron reaction bucket LAW=22
		P_NEU_R_24,      // pointwised neutron reaction bucket LAW=24
		P_NEU_R_44,      // pointwised neutron reaction bucket LAW=44
		P_NEU_R_61,      // pointwised neutron reaction bucket LAW=61
		P_NEU_R_66,      // pointwised neutron reaction bucket LAW=66
		P_NEU_COH,       // pointwised neutron reaction bucket, coherent elastic
		P_NEU_INCOH,     // pointwised neutron reaction bucket, incoherent inelastic
		P_NEU_GAMMA,     // pointwised neutron gamma bucket
		DELTA,           // generic ion delta
		ION_NUCLEAR,     // Nucleus-nucleus inelastic (determine isotope target & rejection)
		// ion-nuclear models
		BME,             // Boltzmann master equation
		QMD_000_032,     // QMD bucket 0
		QMD_033_064,     // QMD bucket 1
		QMD_065_096,     // QMD bucket 2
		QMD_097_128,     // QMD bucket 3
		QMD_129_160,     // QMD bucket 4
		QMD_161_192,     // QMD bucket 5
		QMD_193_224,     // QMD bucket 6
		QMD_225_256,     // QMD bucket 7
		QMD_257_288,     // QMD bucket 8 (cover possible maximum reaction, O-18 + Fm-252)
		CN_FORMATION,    // compound nucleus formation (INCL)
		ABRASION,
		NUC_SECONDARY,   // nuclear model secondaries
		DEEXCITATION,    // de-excitation branching
		PHOTON_EVAP,     // photon evaporation
		COMP_FISSION,    // competitive fission
		// buffer size indicator
		EOB,
	};


	constexpr int BUFFER_ID_QMD_OFFSET = (int)BUFFER_TYPE::QMD_000_032;
	constexpr int BUFFER_ID_QMD_STRIDE = (int)BUFFER_TYPE::QMD_257_288 - BUFFER_ID_QMD_OFFSET + 1;


	constexpr unsigned char BUFFER_ID_P_NEU_OFFSET = (unsigned char)BUFFER_TYPE::P_NEU_R_1;
	constexpr unsigned char BUFFER_ID_P_NEU_STRIDE = (unsigned char)BUFFER_TYPE::P_NEU_INCOH - BUFFER_ID_P_NEU_OFFSET + 1;
	constexpr unsigned char BUFFER_ID_P_NEU_CEIL   = (unsigned char)BUFFER_TYPE::P_NEU_INCOH + 1;


	static const std::map<int, BUFFER_TYPE>& getPidHash() {
		static const std::map<int, BUFFER_TYPE> PID_HASH = {
			{ Define::PID::PID_ELECTRON, BUFFER_TYPE::ELECTRON    },
			{ Define::PID::PID_PHOTON,   BUFFER_TYPE::PHOTON      },
			{ Define::PID::PID_POSITRON, BUFFER_TYPE::POSITRON    },
			{ Define::PID::PID_NEUTRON,  BUFFER_TYPE::GNEUTRON    },
			{ Define::PID::PID_GENION,   BUFFER_TYPE::GENION      },
		};
		return PID_HASH;
	}
    

	static const std::map<int, std::string>& getPidName() {
		static const std::map<int, std::string> PID_NAME = {
			{ Define::PID::PID_ELECTRON, "Electron"       },
			{ Define::PID::PID_PHOTON,   "Photon"         },
			{ Define::PID::PID_POSITRON, "Positron"       },
			{ Define::PID::PID_NEUTRON,  "Neutron"        },
			{ Define::PID::PID_GENION,   "Generic Ion"    },
		};
		return PID_NAME;
	}


	class BufferProperty {
	private:
		std::string _name;
		size_t      _importance;
		bool        _use_za_data;
		std::set<BUFFER_TYPE> _parent_kernel;
		BUFFER_KERNEL_LAUNCH_STRUCTURE _kernel_structure;
		BUFFER_KERNEL_PHYSICS_PROPERTY _kernel_physics;
		

	public:


		BufferProperty() :
			_importance(0u),
			_name(""),
			_use_za_data(false),
			_kernel_structure(BUFFER_KERNEL_LAUNCH_STRUCTURE::VIRTUAL),
			_kernel_physics(BUFFER_KERNEL_PHYSICS_PROPERTY::SOURCE) {}


		BufferProperty(
			const std::string& name,
			size_t importance,
			bool use_za,
			BUFFER_KERNEL_LAUNCH_STRUCTURE  kernel_structure,
			BUFFER_KERNEL_PHYSICS_PROPERTY kernel_physics
		) :
			_name(name),
			_importance(importance),
			_use_za_data(use_za),
			_kernel_structure(kernel_structure),
			_kernel_physics(kernel_physics) {}


		void setParentBuffer(BUFFER_TYPE parent) {
			this->_parent_kernel.insert(parent);
		}


		const std::string& name() const { return this->_name; }


		size_t importance() const { return this->_importance; }


		bool useZA() const { return this->_use_za_data; }


		const std::set<BUFFER_TYPE>& parentKernel() const { return this->_parent_kernel; }


		BUFFER_KERNEL_LAUNCH_STRUCTURE structure() const { return this->_kernel_structure; }


		BUFFER_KERNEL_PHYSICS_PROPERTY physicsType() const { return this->_kernel_physics; }


	};


	constexpr size_t IMPORTANCE_DEFAULT             = 2;
	constexpr size_t IMPORTANCE_NON_SHOWER_PARTICLE = 2;
	constexpr size_t IMPORTANCE_SHOWER_PARTICLE     = 20;
	constexpr size_t IMPORTANCE_LOW_ENERGY_NEUTRON  = 8;
	constexpr size_t IMPORTANCE_CASCADE_REACTION    = 8;


	const BufferProperty& getBufferProperty(BUFFER_TYPE btype);


}
