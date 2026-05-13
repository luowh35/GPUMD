/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "ensemble.cuh"
#include "utilities/gpu_macro.cuh"
#include <vector>
#ifdef USE_HIP
  #include <hipfft/hipfft.h>
#else
  #include <cufft.h>
#endif
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif

class Ensemble_QTB : public Ensemble
{
public:
  // NVT-QTB constructor
  Ensemble_QTB(
    int t,
    int N,
    double T,
    double Tc,
    double dt,
    double f_max,
    int N_f,
    bool use_adaptive_qtb,
    bool use_theta_correction,
    bool use_legacy_scheme,
    bool enforce_cutoff,
    double cutoff_taper,
    int adaptive_optimizer,
    double adaptive_rate,
    double adaptive_tau_average,
    double adaptive_tau_adapt,
    double adaptive_smooth_width,
    double adaptive_window,
    const std::vector<double>& adaptive_rate_type);

  ~Ensemble_QTB(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

private:
  int number_of_atoms;
  int number_of_atom_types;
  int N_f;
  int nfreq2;
  int alpha;
  int counter_mu;
  int adaptive_segment_length;
  int time_filter_count;
  int adaptive_sample_count;
  int adaptive_update_count;

  double dt;
  double h_timestep;
  double fric_coef;
  double f_max_natural;
  double last_filter_temperature;
  double adaptive_rate;
  double adaptive_tau_average;
  double adaptive_tau_adapt;
  double adaptive_smooth_width;
  double adaptive_window;
  double adaptive_gamma_min;
  double adaptive_gamma_max;
  double legacy_force_prefactor;

  bool use_adaptive_qtb;
  bool use_theta_correction;
  bool use_legacy_scheme;
  bool enforce_cutoff;
  int adaptive_optimizer;
  double cutoff_taper;
  bool filter_is_dirty;
  bool adaptive_qtb_initialized;
  bool adaptive_fft_plan_initialized;
  bool adaptive_gamma_floor_warning_issued;
  bool legacy_fran_ready;

  std::vector<double> time_H_host;
  std::vector<int> atom_type_counts_host;
  std::vector<double> atom_type_masses_host;
  std::vector<double> adaptive_rate_type_host;
  std::vector<double> gamma_spectrum_host;
  std::vector<double> gamma_initial_spectrum_host;
  std::vector<double> adaptive_vv_host;
  std::vector<double> adaptive_vr_raw_host;
  std::vector<double> adaptive_vr_host;
  std::vector<double> adaptive_ff_host;
  std::vector<double> adaptive_vv_average_host;
  std::vector<double> adaptive_vr_average_host;
  std::vector<double> adaptive_gamma_running_host;
  std::vector<double> adaptive_vv_segment_host;
  std::vector<double> adaptive_vr_segment_raw_host;
  std::vector<double> adaptive_ff_segment_host;
  std::vector<int> adaptive_average_count_host;
  std::vector<int> adaptive_gamma_count_host;
  GPU_Vector<double> time_H_device;
  GPU_Vector<double> random_array_0;
  GPU_Vector<double> random_array_1;
  GPU_Vector<double> random_array_2;
  GPU_Vector<double> fran;
  GPU_Vector<gpurandState> curand_states;
  GPU_Vector<double> adaptive_vv_sums;
  GPU_Vector<double> adaptive_vr_sums;
  GPU_Vector<double> adaptive_ff_sums;
  GPU_Vector<gpufftComplex> adaptive_velocity_history;
  GPU_Vector<gpufftComplex> adaptive_random_history;
  gpufftHandle adaptive_fft_plan;
  FILE* adaptive_gamma_file;
  FILE* adaptive_fdt_file;
  FILE* adaptive_theta_file;
  FILE* legacy_debug_file;
  double last_theta_dump_temperature;

  void init_qtb_common(
    int N,
    double T,
    double Tc,
    double dt_input,
    double f_max_input,
    int N_f_input,
    bool use_adaptive_qtb_input,
    bool use_theta_correction_input,
    bool use_legacy_scheme_input,
    bool enforce_cutoff_input,
    double cutoff_taper_input,
    int adaptive_optimizer_input,
    double adaptive_rate_input,
    double adaptive_tau_average_input,
    double adaptive_tau_adapt_input,
    double adaptive_smooth_width_input,
    double adaptive_window_input,
    const std::vector<double>& adaptive_rate_type_input);
  void initialize_adaptive_qtb();
  bool load_adaptive_gamma_restart();
  void write_adaptive_gamma_restart();
  void write_adaptive_qtb_diagnostics();
  void write_theta_correction_diagnostics(
    const double target_temperature,
    const std::vector<double>& raw_theta,
    const std::vector<double>& corrected_theta);
  void update_time_filter(const double target_temperature);
  void sample_adaptive_qtb();
  void sample_adaptive_qtb_ou_center();
  void adapt_random_force_spectrum();
  void refresh_colored_random_force(const bool shift_history = true);
  void apply_legacy_qtb_force(const std::vector<Group>& group);
  void write_legacy_qtb_debug(const std::vector<Group>& group);
  void apply_qtb_force_half_step(const std::vector<Group>& group);
  void apply_qtb_position_half_step(const std::vector<Group>& group);
  void apply_qtb_ou_step();
};
