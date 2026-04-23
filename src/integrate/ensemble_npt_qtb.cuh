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

#include "ensemble_mttk.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstdio>
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

class Ensemble_NPT_QTB : public Ensemble_MTTK
{
public:
  Ensemble_NPT_QTB(const char** params, int num_params);
  virtual ~Ensemble_NPT_QTB(void);

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
  int qtb_number_of_atoms;
  int qtb_number_of_atom_types;
  int qtb_N_f;
  int qtb_nfreq2;
  int qtb_alpha;
  int qtb_counter_mu;
  int qtb_adaptive_segment_length;
  int qtb_time_filter_count;
  int qtb_adaptive_sample_count;
  int qtb_adaptive_update_count;

  double qtb_dt;
  double qtb_h_timestep;
  double qtb_fric_coef;
  double qtb_f_max_natural;
  double qtb_last_filter_temperature;
  double qtb_adaptive_rate;
  double qtb_adaptive_window;
  double qtb_adaptive_gamma_min;
  double qtb_adaptive_gamma_max;
  double qtb_f_max = 200.0;

  std::vector<double> qtb_time_H_host;
  std::vector<int> qtb_atom_type_counts_host;
  std::vector<double> qtb_atom_type_masses_host;
  std::vector<double> qtb_adaptive_rate_type_host;
  std::vector<double> qtb_gamma_spectrum_host;
  std::vector<double> qtb_adaptive_vv_host;
  std::vector<double> qtb_adaptive_vr_raw_host;
  std::vector<double> qtb_adaptive_vr_host;
  std::vector<double> qtb_adaptive_ff_host;
  std::vector<double> qtb_adaptive_vv_segment_host;
  std::vector<double> qtb_adaptive_vr_segment_raw_host;
  std::vector<double> qtb_adaptive_ff_segment_host;
  GPU_Vector<double> qtb_time_H_device;
  GPU_Vector<double> qtb_random_array_0;
  GPU_Vector<double> qtb_random_array_1;
  GPU_Vector<double> qtb_random_array_2;
  GPU_Vector<double> qtb_fran;
  GPU_Vector<gpurandState> qtb_curand_states;
  GPU_Vector<double> qtb_adaptive_vv_sums;
  GPU_Vector<double> qtb_adaptive_vr_sums;
  GPU_Vector<double> qtb_adaptive_ff_sums;
  GPU_Vector<gpufftComplex> qtb_adaptive_velocity_history;
  GPU_Vector<gpufftComplex> qtb_adaptive_random_history;
  gpufftHandle qtb_adaptive_fft_plan;
  FILE* qtb_adaptive_gamma_file;
  FILE* qtb_adaptive_fdt_file;
  FILE* qtb_adaptive_theta_file;
  double qtb_last_theta_dump_temperature;

  bool qtb_use_adaptive;
  bool qtb_filter_is_dirty;
  bool qtb_adaptive_initialized;
  bool qtb_adaptive_fft_plan_initialized;
  bool qtb_adaptive_gamma_floor_warning_issued;

  void init_qtb();
  void initialize_adaptive_qtb();
  void write_adaptive_qtb_diagnostics();
  void write_theta_correction_diagnostics(
    const double target_temperature,
    const std::vector<double>& raw_theta,
    const std::vector<double>& corrected_theta);
  void qtb_update_time_filter(const double target_temperature);
  void sample_adaptive_qtb();
  void adapt_random_force_spectrum();
  void qtb_refresh_colored_random_force();
  void qtb_apply_half_step();

protected:
  virtual void init_mttk() override;
  virtual void get_target_temp() override;
};
