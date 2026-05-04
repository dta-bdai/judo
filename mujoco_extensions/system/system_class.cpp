// Copyright (c) 2024-2025 Boston Dynamics AI Institute LLC. All rights reserved.
#include <chrono>

#include "system/system_class.h"

namespace SystemClass {

namespace {
constexpr int kExpectedObservationSize = 84;
constexpr int kExpectedCommandSize = 25;
constexpr int kExpectedPolicyOutputSize = 12;

void validatePolicySchema(OnnxInterface::Policy& policy, const VectorT& observation) {
  if (observation.size() != kExpectedObservationSize) {
    throw std::runtime_error("Unexpected observation buffer size; expected 84");
  }
  if (policy.input_size != observation.size()) {
    throw std::runtime_error("Policy input size mismatch: expected 84-dimensional observation input");
  }
  if (policy.output_size != kExpectedPolicyOutputSize) {
    throw std::runtime_error("Policy output size mismatch: expected 12-dimensional action output");
  }
  if (policy.getInputName() != "obs") {
    throw std::runtime_error("Unexpected ONNX input name; expected 'obs'");
  }
  if (policy.getOutputName() != "actions") {
    throw std::runtime_error("Unexpected ONNX output name; expected 'actions'");
  }
}
}  // namespace

// Constructor
System::System(const std::string& model_filepath_, const std::string& policy_filepath_)
    : model_filepath(model_filepath_),
      policy_filepath(policy_filepath_),
      observation(3 + 3 + 3 + 3 + 7 + 12 + 3 + 19 + 19 + 12),  // 84
      policy_output(1),
      policy_input_(1),
      control_(19),
      default_joint_pos_(19),
      joint_pos_(19),
      joint_vel_(19) {
  // Initialize the permutation matrices
  initializeSystemIndices();
  zeroOutVectors();

  model = SystemUtils::loadModel(model_filepath);
  data = mj_makeData(model);
  setStateIndices();
  loadPolicy(policy_filepath, OnnxInterface::getNullSession());

  policy_input_.resize(policy.input_size);
  policy_output.resize(policy.output_size);
  policy_input_.setZero();
  policy_output.setZero();
  validatePolicySchema(policy, observation);
}

System::System(const std::string& policy_filepath_, const mjModel* reference_model,
               std::shared_ptr<OnnxInterface::Session> reference_session)
    : policy_filepath(policy_filepath_),
      observation(3 + 3 + 3 + 3 + 7 + 12 + 3 + 19 + 19 + 12),  // 84
      policy_output(1),
      policy_input_(1),
      control_(19),
      default_joint_pos_(19),
      joint_pos_(19),
      joint_vel_(19) {
  // Initialize the permutation matrices
  initializeSystemIndices();
  zeroOutVectors();

  model = mj_copyModel(nullptr, reference_model);
  if (!model) {
    throw std::runtime_error("Failed to load copy XML file");
  }
  data = mj_makeData(model);
  setStateIndices();

  loadPolicy(policy_filepath, reference_session);
  policy_input_.resize(policy.input_size);
  policy_output.resize(policy.output_size);
  policy_input_.setZero();
  policy_output.setZero();
  validatePolicySchema(policy, observation);
}

System::~System() {
  if (data) {
    mj_deleteData(data);
  }
  if (model) {
    mj_deleteModel(model);
  }
}

void System::reset(const bool reset_last_output) {
  mj_resetData(model, data);
  if (reset_last_output) {
    policy_output.setZero();
  }
  for (int i = 0; i < model->nsensordata; i++) {
    data->sensordata[i] = 0.0;
  }
}

void System::zeroOutVectors() {
  observation.setZero();
  control_.setZero();
  policy_output.setZero();
  joint_pos_.setZero();
  joint_vel_.setZero();
}

void System::setStateIndices() {
  // Try Starfish-style namespaced names first, then fall back to the bare names
  // used by Judo's XMLs. Throws if neither is found so we never silently index
  // into garbage memory via mj_name2id returning -1.
  auto resolveBody = [&](std::initializer_list<const char*> candidates) -> int {
    for (const char* name : candidates) {
      const int id = mj_name2id(model, mjOBJ_BODY, name);
      if (id >= 0) {
        return id;
      }
    }
    std::string msg = "[FATAL] None of the expected Spot body names were found in the model:";
    for (const char* name : candidates) {
      msg += " '";
      msg += name;
      msg += "'";
    }
    throw std::runtime_error(msg);
  };

  const int base_id = resolveBody({"spot/torso", "torso", "body"});            // id of the torso
  base_qpos_start_idx = model->jnt_qposadr[model->body_jntadr[base_id]];       // base position address
  base_qvel_start_idx = model->jnt_dofadr[model->body_jntadr[base_id]];        // base velocity address
  const int first_leg_id = resolveBody({"spot/front_left_hip", "front_left_hip"});  // id of the first leg
  leg_qpos_start_idx = model->jnt_qposadr[model->body_jntadr[first_leg_id]];   // leg position address
  leg_qvel_start_idx = model->jnt_dofadr[model->body_jntadr[first_leg_id]];    // leg velocity address
}

void System::loadPolicy(const std::string& policy_filepath_,
                        std::shared_ptr<OnnxInterface::Session> reference_session) {
  if (reference_session.get() == nullptr) {
    policy = OnnxInterface::Policy(policy_filepath_);
  } else {
    policy = OnnxInterface::Policy(policy_filepath_, reference_session);
  }
}

void System::initializeSystemIndices() {
  Eigen::ArrayXi orbit_to_mujoco_legs_indices(12);
  orbit_to_mujoco_legs_indices << 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11;
  orbit_to_mujoco_legs.indices() = orbit_to_mujoco_legs_indices;

  Eigen::ArrayXi mujoco_to_orbit_legs_indices(12);
  mujoco_to_orbit_legs_indices << 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11;
  mujoco_to_orbit_legs.indices() = mujoco_to_orbit_legs_indices;

  Eigen::ArrayXi orbit_to_mujoco_indices(19);
  orbit_to_mujoco_indices << 12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18;
  orbit_to_mujoco.indices() = orbit_to_mujoco_indices;

  Eigen::ArrayXi mujoco_to_orbit_indices(19);
  mujoco_to_orbit_indices << 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18;
  mujoco_to_orbit.indices() = mujoco_to_orbit_indices;

  default_joint_pos_ << 0.12, 0.5, -1, -0.12, 0.5, -1, 0.12, 0.5, -1, -0.12, 0.5, -1,  // legs pos (mujoco)
      0, -0.9, 1.8, 0, -0.9, 0, -1.54;                                                 // arm pos (mujoco)
}

// Compute observation
void System::setObservation(const VectorT& command) {
  if (command.size() != kExpectedCommandSize) {
    throw std::runtime_error("Command size mismatch: expected 25-dimensional locomotion command");
  }

  VectorT torso_vel_command = command.segment(0, 3);
  VectorT arm_command = command.segment(3, 7);
  VectorT leg_command = command.segment(10, 12);
  VectorT torso_pos_command = command.segment(22, 3);

  // Initialization
  int offset = 0;
  double inv_base_quat[4];
  double base_linear_velocity[3];
  double base_angular_velocity[3];
  double projected_gravity[3] = {0, 0, -1.0};

  // Indices
  const int base_quat_start_index = this->base_qpos_start_idx + 3;
  const int base_linvel_start_index = this->base_qvel_start_idx;
  const int base_angvel_start_index = this->base_qvel_start_idx + 3;
  const int joint_pos_start_index = this->leg_qpos_start_idx;
  const int joint_vel_start_index = this->leg_qvel_start_idx;

  // Compute observation data
  mju_negQuat(inv_base_quat, data->qpos + base_quat_start_index);

  mju_rotVecQuat(base_linear_velocity, data->qvel + base_linvel_start_index, inv_base_quat);

  for (int i = 0; i < 3; ++i) {
    base_angular_velocity[i] = data->qvel[base_angvel_start_index + i];
  }

  mju_rotVecQuat(projected_gravity, projected_gravity, inv_base_quat);

  for (int i = 0; i < 19; i++) {
    joint_pos_[i] = data->qpos[joint_pos_start_index + i];
    joint_vel_[i] = data->qvel[joint_vel_start_index + i];
  }

  // Populate the observation vector
  for (int i = 0; i < 3; i++) {
    observation[offset + i] = base_linear_velocity[i];
  }
  offset += 3;

  for (int i = 0; i < 3; i++) {
    observation[offset + i] = base_angular_velocity[i];
  }
  offset += 3;

  for (int i = 0; i < 3; i++) {
    observation[offset + i] = projected_gravity[i];
  }
  offset += 3;

  for (int i = 0; i < 3; i++) {
    observation[offset + i] = torso_vel_command[i];
  }
  offset += 3;
  for (int i = 0; i < 7; i++) {
    observation[offset + i] = arm_command[i];
  }
  offset += 7;

  for (int i = 0; i < 12; i++) {
    observation[offset + i] = leg_command[i];
  }
  offset += 12;

  for (int i = 0; i < 3; i++) {
    observation[offset + i] = torso_pos_command[i];
  }
  offset += 3;

  for (int i = 0; i < 19; i++) {
    observation[offset + i] = joint_pos_[i];
  }
  offset += 19;

  for (int i = 0; i < 19; i++) {
    observation[offset + i] = joint_vel_[i];
  }
  offset += 19;

  for (int i = 0; i < 12; i++) {
    observation[offset + i] = policy_output[i];
  }
  offset += 12;
}

// evaluate neural network policy
void System::policyInference() {
  if (policy.input_size != observation.size()) {
    throw std::runtime_error("Policy input size does not match observation size");
  }
  if (policy_output.size() != kExpectedPolicyOutputSize) {
    throw std::runtime_error("Policy output size does not match expected leg action size (12)");
  }

  // TODO(@bhung) Get most of these hard coded index values from another source
  // Convert observation to float and initialize torch_input
  for (int i = 0; i < this->policy.input_size; ++i) {
    this->policy_input_[i] = static_cast<float>(this->observation[i]);
  }
  OnnxInterface::VectorT policy_input_float_ = policy_input_.cast<float>();
  OnnxInterface::VectorT policy_output_float_ = this->policy.policyInference(&policy_input_float_);
  this->policy_output = policy_output_float_.cast<double>();

  // legs (output by the neural net; includes any leg override in policy output)
  control_.segment(0, 12) = policy_output;
  // arm (read neural net input i.e. from observation)
  control_.segment(12, 7) = observation.segment(3 + 3 + 3 + 3, 7);

  for (int i = 0; i < 19; i++) {
    data->ctrl[i] = control_[i];
  }
}

// extract the control vector out of the mj Data
VectorT System::getControl() {
  const int nu = model->nu;
  VectorT control(nu);
  for (int i = 0; i < nu; i++) {
    control[i] = data->ctrl[i];
  }
  return control;
}

// extracts the state vector out of the mj model
VectorT System::getState() {
  int nq = model->nq;
  int nv = model->nv;
  VectorT state_vec(nq + nv);
  for (int i = 0; i < nq; ++i) {
    state_vec[i] = data->qpos[i];
  }
  for (int i = 0; i < nv; ++i) {
    state_vec[nq + i] = data->qvel[i];
  }
  return state_vec;
}

std::tuple<MatrixT, MatrixT> System::rollout(const VectorT& state, const MatrixT& command, const int physics_substeps,
                                             const bool reset_last_output, const double cutoff_time) {
  // Resetting mujoco data, last output and sensordata to zero
  this->reset(reset_last_output);

  // Initialization: return num_commands states (one per command, sampled at end of substeps)
  const auto num_commands = command.rows();
  const int nq = model->nq;
  const int nv = model->nv;
  const int nsensordata = model->nsensordata;
  MatrixT states = MatrixT::Zero(num_commands, nq + nv);
  MatrixT sensors = MatrixT::Zero(num_commands, nsensordata);
  SystemUtils::setState(model, data, state);

  // Rollout physics and policy
  double seconds_elapsed = 0.0;
  auto rollout_start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_commands; ++i) {
    // Check cutoff time at start of each command
    auto loop_start_time = std::chrono::high_resolution_clock::now();
    seconds_elapsed = std::chrono::duration<double>(loop_start_time - rollout_start_time).count();

    if (seconds_elapsed < cutoff_time) {
      // Policy inference ONCE per command
      this->setObservation(command.row(i));
      this->System::policyInference();

      // Apply same control for all physics substeps
      for (int k = 0; k < physics_substeps; ++k) {
        mj_step(model, data);
      }

      // Record state at end of substeps
      for (int l = 0; l < nq; ++l) {
        states(i, l) = data->qpos[l];
      }
      for (int l = 0; l < nv; ++l) {
        states(i, nq + l) = data->qvel[l];
      }
      for (int l = 0; l < nsensordata; ++l) {
        sensors(i, l) = data->sensordata[l];
      }
    } else {
      // Fill in the rest of the state with the last calculated state
      int prev_ind = std::max(i - 1, 0);
      for (int l = 0; l < nq; ++l) {
        states(i, l) = states(prev_ind, l);
      }
      for (int l = 0; l < nv; ++l) {
        states(i, nq + l) = states(prev_ind, nq + l);
      }
      for (int l = 0; l < nsensordata; ++l) {
        sensors(i, l) = sensors(prev_ind, l);
      }
    }
  }

  return {states, sensors};
}

std::tuple<MatrixTList, MatrixTList, VectorTList> threadedRollout(const std::vector<std::shared_ptr<System>>& systems,
                                                                  const VectorTList& states, const MatrixTList& command,
                                                                  const VectorTList& last_policy_output,
                                                                  const int num_threads, const int physics_substeps,
                                                                  const double cutoff_time) {
  int effective_threads = std::min(static_cast<int>(systems.size()), num_threads);

  // Vector to store threads
  std::vector<std::thread> threads;
  MatrixTList output_states(effective_threads);
  MatrixTList sensors(effective_threads);
  VectorTList policy_outputs(effective_threads);

  // Start threads
  for (int i = 0; i < effective_threads; ++i) {
    threads.emplace_back([i, &systems, &states, &command, &last_policy_output, &output_states, &sensors,
                          physics_substeps, cutoff_time]() {
      systems[i]->policy_output = last_policy_output[i];
      std::tie(output_states[i], sensors[i]) =
          systems[i]->rollout(states[i], command[i], physics_substeps, false, cutoff_time);
    });
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Update last_policy_output after all threads have completed
  for (int i = 0; i < effective_threads; ++i) {
    policy_outputs[i] = systems[i]->policy_output;
  }

  return {output_states, sensors, policy_outputs};
}

std::vector<std::shared_ptr<SystemClass::System>> create_systems_vector(const mjModel* reference_model,
                                                                        const std::string& policy_filepath,
                                                                        const int num_systems) {
  // Vector to store the systems
  std::vector<std::shared_ptr<SystemClass::System>> systems(num_systems);

  std::shared_ptr<OnnxInterface::Session> reference_session =
      std::shared_ptr<OnnxInterface::Session>(OnnxInterface::allocateOrtSession(policy_filepath));
  // Function to create a system in a separate thread
  auto create_system = [&](int i) {
    systems[i] = std::make_shared<SystemClass::System>(policy_filepath, reference_model, reference_session);
  };

  // Vector to hold the threads
  std::vector<std::thread> threads;
  threads.reserve(num_systems);

  // Launch threads to create System instances in parallel
  for (int i = 0; i < num_systems; i++) {
    threads.emplace_back(create_system, i);
  }

  // Wait for all threads to finish
  for (auto& thread : threads) {
    thread.join();
  }

  return systems;
}

}  // namespace SystemClass
