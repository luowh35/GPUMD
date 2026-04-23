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

/*----------------------------------------------------------------------------80
NPT-QTB: Parrinello-Rahman barostat (MTTK) with QTB colored noise thermostat.
Equivalent to LAMMPS fix nph + fix qtb.
[1] Dammak, T., et al. Phys. Rev. Lett. 103, 190601 (2009).
[2] Martyna, G. J., et al. J. Chem. Phys. 101, 4177 (1994).
------------------------------------------------------------------------------*/

#include "ensemble_npt_qtb.cuh"
#include "langevin_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstring>
#include <iostream>

namespace
{
static int get_gamma_index_from_fft_bin(const int N_f, const int fft_bin)
{
  if (fft_bin == 0) {
    return N_f;
  }
  if (fft_bin == N_f) {
    return 0;
  }
  return N_f + fft_bin;
}

static double get_qtb_energy_density(const double frequency, const double temperature)
{
  if (frequency <= 0.0) {
    return K_B * temperature;
  }

  const double energy = 2.0 * PI * HBAR * frequency;
  const double x = energy / (K_B * temperature);
  double qfactor = 0.5;
  if (x < 200.0) {
    qfactor += 1.0 / (exp(x) - 1.0);
  }
  return energy * qfactor;
}

static double get_ou_spectrum_correction(
  const double omega,
  const double fric_coef,
  const double delta_t)
{
  if (delta_t <= 0.0) {
    return 1.0;
  }

  const double exp_factor = exp(-fric_coef * delta_t);
  const double numerator =
    1.0 - 2.0 * exp_factor * cos(omega * delta_t) + exp_factor * exp_factor;
  const double denominator =
    (fric_coef * fric_coef + omega * omega) * delta_t * delta_t;

  if (denominator < 1.0e-30) {
    if (fric_coef < 1.0e-15) {
      return 1.0;
    }
    const double one_minus_exp = 1.0 - exp_factor;
    return one_minus_exp * one_minus_exp / (fric_coef * fric_coef * delta_t * delta_t);
  }

  return numerator / denominator;
}

static double interpolate_adaptive_spectrum(
  const std::vector<double>& spectrum,
  const int segment_length,
  const int type_offset,
  const double fft_bin)
{
  const int max_bin = segment_length / 2;
  if (max_bin <= 0 || fft_bin <= 0.0) {
    return spectrum[type_offset];
  }
  if (fft_bin >= max_bin) {
    return spectrum[type_offset + max_bin];
  }

  const int left_bin = int(floor(fft_bin));
  const int right_bin = left_bin + 1;
  const double fraction = fft_bin - left_bin;
  return spectrum[type_offset + left_bin] * (1.0 - fraction) +
         spectrum[type_offset + right_bin] * fraction;
}

static double interpolate_uniform_spectrum(
  const std::vector<double>& spectrum,
  const double spacing,
  const double coordinate)
{
  const int max_index = int(spectrum.size()) - 1;
  if (max_index <= 0 || coordinate <= 0.0) {
    return spectrum[0];
  }
  if (spacing <= 0.0) {
    return spectrum[max_index];
  }

  const double grid_index = coordinate / spacing;
  if (grid_index >= max_index) {
    return spectrum[max_index];
  }

  const int left_index = int(floor(grid_index));
  const int right_index = left_index + 1;
  const double fraction = grid_index - left_index;
  return spectrum[left_index] * (1.0 - fraction) +
         spectrum[right_index] * fraction;
}

static void compute_corrected_qtb_energy_density(
  const int N_f,
  const int nfreq2,
  const double h_timestep,
  const double fric_coef,
  const double temperature,
  std::vector<double>& raw_theta,
  std::vector<double>& corrected_theta)
{
  corrected_theta.resize(size_t(N_f) + 1);
  corrected_theta[0] = K_B * temperature;
  if (N_f <= 0) {
    raw_theta.resize(1);
    raw_theta[0] = corrected_theta[0];
    return;
  }

  const double domega = 2.0 * PI / (nfreq2 * h_timestep);
  const double epsilon = 1.0e-30;
  const int num_iterations = 50;
  const int auxiliary_density = 2;
  const double auxiliary_extent = 1.5;
  int auxiliary_positive_size = int(ceil(auxiliary_extent * auxiliary_density * N_f));
  if (auxiliary_positive_size < N_f) {
    auxiliary_positive_size = N_f;
  }
  const double auxiliary_domega = domega / auxiliary_density;

  std::vector<double> omega_grid(size_t(N_f) + 1, 0.0);
  raw_theta.resize(size_t(N_f) + 1, 0.0);
  std::vector<double> auxiliary_raw_theta(size_t(auxiliary_positive_size) + 1, 0.0);
  std::vector<double> auxiliary_corrected_theta(size_t(auxiliary_positive_size) + 1, 0.0);
  std::vector<double> target(size_t(auxiliary_positive_size), 0.0);
  std::vector<double> kernel(
    size_t(auxiliary_positive_size) * size_t(auxiliary_positive_size), 0.0);
  std::vector<double> normalization(size_t(auxiliary_positive_size), 0.0);
  std::vector<double> estimate(size_t(auxiliary_positive_size), 0.0);
  std::vector<double> estimate_new(size_t(auxiliary_positive_size), 0.0);
  std::vector<double> blurred(size_t(auxiliary_positive_size), 0.0);
  std::vector<double> ratio(size_t(auxiliary_positive_size), 0.0);

  for (int j = 0; j <= N_f; ++j) {
    const double frequency = j / (nfreq2 * h_timestep);
    omega_grid[j] = 2.0 * PI * frequency;
    raw_theta[j] = get_qtb_energy_density(frequency, temperature);
  }
  auxiliary_corrected_theta[0] = corrected_theta[0];

  for (int j = 0; j <= auxiliary_positive_size; ++j) {
    const double frequency = j * auxiliary_domega / (2.0 * PI);
    auxiliary_raw_theta[j] = get_qtb_energy_density(frequency, temperature);
  }

  for (int i = 1; i <= auxiliary_positive_size; ++i) {
    const double omega_i = i * auxiliary_domega;
    const double omega_i_sq = omega_i * omega_i;
    const double kernel_zero = fric_coef / (PI * omega_i_sq);
    double rhs =
      0.5 * auxiliary_raw_theta[i] -
      kernel_zero * 0.5 * auxiliary_domega * auxiliary_corrected_theta[0];
    if (rhs < epsilon) {
      rhs = 0.5 * auxiliary_raw_theta[i];
    }
    target[i - 1] = rhs;

    for (int j = 1; j <= auxiliary_positive_size; ++j) {
      const double omega_j = j * auxiliary_domega;
      const double omega_j_sq = omega_j * omega_j;
      const double denom =
        (omega_j_sq - omega_i_sq) * (omega_j_sq - omega_i_sq) +
        fric_coef * fric_coef * omega_j_sq;
      double weight = auxiliary_domega;
      if (j == auxiliary_positive_size) {
        weight *= 0.5;
      }
      const double kernel_value = fric_coef * omega_i_sq * weight / (PI * denom);
      kernel[(i - 1) * auxiliary_positive_size + (j - 1)] = kernel_value;
      normalization[j - 1] += kernel_value;
    }
  }

  for (int j = 0; j < auxiliary_positive_size; ++j) {
    estimate[j] = auxiliary_raw_theta[j + 1];
    if (normalization[j] < epsilon) {
      normalization[j] = epsilon;
    }
  }

  for (int iter = 0; iter < num_iterations; ++iter) {
    for (int i = 0; i < auxiliary_positive_size; ++i) {
      double blurred_value = 0.0;
      for (int j = 0; j < auxiliary_positive_size; ++j) {
        blurred_value += kernel[i * auxiliary_positive_size + j] * estimate[j];
      }
      if (blurred_value < epsilon) {
        blurred_value = epsilon;
      }
      blurred[i] = blurred_value;
      ratio[i] = target[i] / blurred_value;
    }

    for (int j = 0; j < auxiliary_positive_size; ++j) {
      double update_factor = 0.0;
      for (int i = 0; i < auxiliary_positive_size; ++i) {
        update_factor += kernel[i * auxiliary_positive_size + j] * ratio[i];
      }
      estimate_new[j] = estimate[j] * update_factor / normalization[j];
      if (estimate_new[j] < epsilon) {
        estimate_new[j] = epsilon;
      }
    }
    estimate.swap(estimate_new);
  }

  for (int j = 1; j <= auxiliary_positive_size; ++j) {
    auxiliary_corrected_theta[j] = estimate[j - 1];
  }

  for (int j = 1; j <= N_f; ++j) {
    corrected_theta[j] =
      interpolate_uniform_spectrum(auxiliary_corrected_theta, auxiliary_domega, omega_grid[j]);
  }

  if (N_f >= 4 && domega > 0.0 && fric_coef > 0.0) {
    int blend_end = int(ceil(fric_coef / domega)) + 1;
    if (blend_end < 4) {
      blend_end = 4;
    }
    if (blend_end > N_f) {
      blend_end = N_f;
    }

    const double anchor_value = fmax(corrected_theta[blend_end], epsilon);
    for (int k = 1; k < blend_end; ++k) {
      const double t = double(k) / double(blend_end);
      const double weight = t * t * (3.0 - 2.0 * t);
      corrected_theta[k] =
        fmax(epsilon, raw_theta[k] + weight * (anchor_value - raw_theta[k]));
    }
  }
}

static __global__ void gpu_initialize_qtb_history(
  gpurandState* states,
  const int N,
  const int nfreq2,
  double* random_array_0,
  double* random_array_1,
  double* random_array_2)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = states[n];
    const int offset = n * nfreq2;
    for (int m = 0; m < nfreq2; ++m) {
      random_array_0[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
      random_array_1[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
      random_array_2[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
    }
    states[n] = state;
  }
}

static __global__ void gpu_refresh_qtb_random_force(
  gpurandState* states,
  const int N,
  const int nfreq2,
  const int time_filter_count,
  const int* atom_type,
  const double* time_H,
  const double* mass,
  double* random_array_0,
  double* random_array_1,
  double* random_array_2,
  double* fran_x,
  double* fran_y,
  double* fran_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = states[n];
    const int offset = n * nfreq2;
    const int filter_offset = (time_filter_count == 1 ? 0 : atom_type[n]) * nfreq2;

    for (int m = 0; m < nfreq2 - 1; ++m) {
      random_array_0[offset + m] = random_array_0[offset + m + 1];
      random_array_1[offset + m] = random_array_1[offset + m + 1];
      random_array_2[offset + m] = random_array_2[offset + m + 1];
    }
    random_array_0[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);
    random_array_1[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);
    random_array_2[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (int m = 0; m < nfreq2; ++m) {
      const int reverse_index = offset + nfreq2 - m - 1;
      const double h = time_H[filter_offset + m];
      sum_x += h * random_array_0[reverse_index];
      sum_y += h * random_array_1[reverse_index];
      sum_z += h * random_array_2[reverse_index];
    }

    const double sqrt_mass = sqrt(mass[n]);
    fran_x[n] = sum_x * sqrt_mass;
    fran_y[n] = sum_y * sqrt_mass;
    fran_z[n] = sum_z * sqrt_mass;

    states[n] = state;
  }
}

static __global__ void gpu_apply_qtb_ou_step(
  const int N,
  const double c1,
  const double c2,
  const double* mass,
  const double* fran_x,
  const double* fran_y,
  const double* fran_z,
  double* vx,
  double* vy,
  double* vz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double mass_inv = 1.0 / mass[n];
    vx[n] = c1 * vx[n] + c2 * fran_x[n] * mass_inv;
    vy[n] = c1 * vy[n] + c2 * fran_y[n] * mass_inv;
    vz[n] = c1 * vz[n] + c2 * fran_z[n] * mass_inv;
  }
}

static __global__ void gpu_store_qtb_sample(
  const int N,
  const int sample_index,
  const int segment_length,
  const double* vx,
  const double* vy,
  const double* vz,
  const double* fran_x,
  const double* fran_y,
  const double* fran_z,
  gpufftComplex* velocity_history,
  gpufftComplex* random_history)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const int offset_x = n * segment_length + sample_index;
    const int offset_y = (n + N) * segment_length + sample_index;
    const int offset_z = (n + 2 * N) * segment_length + sample_index;

    velocity_history[offset_x].x = float(vx[n]);
    velocity_history[offset_x].y = 0.0f;
    velocity_history[offset_y].x = float(vy[n]);
    velocity_history[offset_y].y = 0.0f;
    velocity_history[offset_z].x = float(vz[n]);
    velocity_history[offset_z].y = 0.0f;

    random_history[offset_x].x = float(fran_x[n]);
    random_history[offset_x].y = 0.0f;
    random_history[offset_y].x = float(fran_y[n]);
    random_history[offset_y].y = 0.0f;
    random_history[offset_z].x = float(fran_z[n]);
    random_history[offset_z].y = 0.0f;
  }
}

static __global__ void gpu_zero_qtb_spectrum_sums(
  const int size,
  double* adaptive_vv_sums,
  double* adaptive_vr_sums,
  double* adaptive_ff_sums)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    adaptive_vv_sums[i] = 0.0;
    adaptive_vr_sums[i] = 0.0;
    adaptive_ff_sums[i] = 0.0;
  }
}

static __global__ void gpu_accumulate_qtb_spectra(
  const int N,
  const int segment_length,
  const int* atom_type,
  const double* mass,
  const gpufftComplex* velocity_history,
  const gpufftComplex* random_history,
  double* adaptive_vv_sums,
  double* adaptive_vr_sums,
  double* adaptive_ff_sums)
{
  const int dof = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_dofs = N * 3;
  if (dof < total_dofs) {
    const int atom_index = dof % N;
    const int type = atom_type[atom_index];
    const double atom_mass = mass[atom_index];
    const int offset = dof * segment_length;
    for (int k = 0; k <= segment_length / 2; ++k) {
      const gpufftComplex velocity_value = velocity_history[offset + k];
      const gpufftComplex random_value = random_history[offset + k];
      const double v_real = velocity_value.x;
      const double v_imag = velocity_value.y;
      const double r_real = random_value.x;
      const double r_imag = random_value.y;
      const double vv = atom_mass * (v_real * v_real + v_imag * v_imag);
      const double vr = v_real * r_real + v_imag * r_imag;
      const double ff = r_real * r_real + r_imag * r_imag;
      atomicAdd(adaptive_vv_sums + type * segment_length + k, vv);
      atomicAdd(adaptive_vr_sums + type * segment_length + k, vr);
      atomicAdd(adaptive_ff_sums + type * segment_length + k, ff);
    }
  }
}

} // namespace

Ensemble_NPT_QTB::Ensemble_NPT_QTB(const char** params, int num_params)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      h[i][j] = h_inv[i][j] = h_old[i][j] = h_old_inv[i][j] = tmp1[i][j] = tmp2[i][j] =
        sigma[i][j] = f_deviatoric[i][j] = p_start[i][j] = p_stop[i][j] = p_current[i][j] =
          p_target[i][j] = p_hydro[i][j] = p_freq[i][j] = omega_dot[i][j] = omega_mass[i][j] =
            p_flag[i][j] = h_ref_inv[i][j] = 0;
      p_period[i][j] = 1000;
      need_scale[i][j] = true;
    }
  }

  ensemble_type = NPH;
  use_barostat = false;
  use_thermostat = false;

  qtb_f_max = 200.0;
  int qtb_n_f_input = 100;
  qtb_use_adaptive = false;
  qtb_adaptive_rate = 0.1;
  qtb_adaptive_window = 0.0;
  qtb_adaptive_gamma_file = nullptr;
  qtb_adaptive_fdt_file = nullptr;
  qtb_adaptive_theta_file = nullptr;
  qtb_adaptive_fft_plan_initialized = false;
  qtb_adaptive_initialized = false;
  qtb_filter_is_dirty = true;
  qtb_adaptive_gamma_floor_warning_issued = false;
  qtb_last_theta_dump_temperature = -1.0;

  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "iso") == 0 || strcmp(params[i], "aniso") == 0 ||
        strcmp(params[i], "tri") == 0) {
      if (i + 2 >= num_params) {
        PRINT_INPUT_ERROR("iso/aniso/tri requires <p_start> <p_stop>.");
      }
      if (!is_valid_real(params[i + 1], &p_start[0][0])) {
        PRINT_INPUT_ERROR("Wrong inputs for p_start.");
      }
      p_start[1][1] = p_start[2][2] = p_start[0][0];
      if (!is_valid_real(params[i + 2], &p_stop[0][0])) {
        PRINT_INPUT_ERROR("Wrong inputs for p_stop.");
      }
      p_stop[1][1] = p_stop[2][2] = p_stop[0][0];
      p_flag[0][0] = p_flag[1][1] = p_flag[2][2] = true;
      use_barostat = true;
      if (strcmp(params[i], "iso") == 0) {
        couple_type = XYZ;
      }
      if (strcmp(params[i], "tri") == 0) {
        for (int a = 0; a < 3; a++) {
          for (int b = 0; b < 3; b++) {
            if (a != b) {
              p_start[a][b] = 0;
              p_stop[a][b] = 0;
              p_flag[a][b] = true;
              need_scale[a][b] = false;
            }
          }
        }
      }
      i += 3;
    } else if (strcmp(params[i], "x") == 0) {
      if (i + 2 >= num_params) {
        PRINT_INPUT_ERROR("x requires <p_start> <p_stop>.");
      }
      if (!is_valid_real(params[i + 1], &p_start[0][0])) {
        PRINT_INPUT_ERROR("Wrong p_start for x.");
      }
      if (!is_valid_real(params[i + 2], &p_stop[0][0])) {
        PRINT_INPUT_ERROR("Wrong p_stop for x.");
      }
      p_flag[0][0] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (strcmp(params[i], "y") == 0) {
      if (i + 2 >= num_params) {
        PRINT_INPUT_ERROR("y requires <p_start> <p_stop>.");
      }
      if (!is_valid_real(params[i + 1], &p_start[1][1])) {
        PRINT_INPUT_ERROR("Wrong p_start for y.");
      }
      if (!is_valid_real(params[i + 2], &p_stop[1][1])) {
        PRINT_INPUT_ERROR("Wrong p_stop for y.");
      }
      p_flag[1][1] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (strcmp(params[i], "z") == 0) {
      if (i + 2 >= num_params) {
        PRINT_INPUT_ERROR("z requires <p_start> <p_stop>.");
      }
      if (!is_valid_real(params[i + 1], &p_start[2][2])) {
        PRINT_INPUT_ERROR("Wrong p_start for z.");
      }
      if (!is_valid_real(params[i + 2], &p_stop[2][2])) {
        PRINT_INPUT_ERROR("Wrong p_stop for z.");
      }
      p_flag[2][2] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (strcmp(params[i], "pperiod") == 0) {
      if (i + 1 >= num_params) {
        PRINT_INPUT_ERROR("pperiod requires a value.");
      }
      if (!is_valid_real(params[i + 1], &p_period[0][0])) {
        PRINT_INPUT_ERROR("Wrong inputs for pperiod.");
      }
      if (p_period[0][0] < 200) {
        PRINT_INPUT_ERROR("pperiod should >= 200 timestep.");
      }
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          p_period[a][b] = p_period[0][0];
        }
      }
      i += 2;
    } else if (strcmp(params[i], "temp") == 0) {
      if (i + 2 >= num_params) {
        PRINT_INPUT_ERROR("temp requires two values: <T_start> <T_stop>.");
      }
      if (!is_valid_real(params[i + 1], &t_start)) {
        PRINT_INPUT_ERROR("Wrong t_start.");
      }
      if (!is_valid_real(params[i + 2], &t_stop)) {
        PRINT_INPUT_ERROR("Wrong t_stop.");
      }
      if (t_start <= 0) {
        PRINT_INPUT_ERROR("t_start should > 0.");
      }
      if (t_stop <= 0) {
        PRINT_INPUT_ERROR("t_stop should > 0.");
      }
      t_target = t_start;
      i += 3;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (i + 1 >= num_params) {
        PRINT_INPUT_ERROR("tperiod requires a value.");
      }
      if (!is_valid_real(params[i + 1], &t_period)) {
        PRINT_INPUT_ERROR("Wrong tperiod.");
      }
      if (t_period <= 0) {
        PRINT_INPUT_ERROR("tperiod should > 0.");
      }
      i += 2;
    } else if (strcmp(params[i], "f_max") == 0) {
      if (i + 1 >= num_params) {
        PRINT_INPUT_ERROR("f_max requires a value.");
      }
      if (!is_valid_real(params[i + 1], &qtb_f_max)) {
        PRINT_INPUT_ERROR("f_max should be a number.");
      }
      if (qtb_f_max <= 0) {
        PRINT_INPUT_ERROR("f_max should > 0.");
      }
      i += 2;
    } else if (strcmp(params[i], "N_f") == 0) {
      if (i + 1 >= num_params) {
        PRINT_INPUT_ERROR("N_f requires a value.");
      }
      if (!is_valid_int(params[i + 1], &qtb_n_f_input)) {
        PRINT_INPUT_ERROR("N_f should be an integer.");
      }
      if (qtb_n_f_input <= 0) {
        PRINT_INPUT_ERROR("N_f should > 0.");
      }
      i += 2;
    } else if (strcmp(params[i], "adaptive") == 0) {
      int adaptive_flag = 0;
      if (i + 1 >= num_params) {
        PRINT_INPUT_ERROR("adaptive requires 0 or 1.");
      }
      if (!is_valid_int(params[i + 1], &adaptive_flag)) {
        PRINT_INPUT_ERROR("adaptive should be 0 or 1.");
      }
      qtb_use_adaptive = (adaptive_flag != 0);
      i += 2;
    } else if (strcmp(params[i], "adapt_rate") == 0) {
      if (i + 1 >= num_params) {
        PRINT_INPUT_ERROR("adapt_rate requires a value.");
      }
      if (!is_valid_real(params[i + 1], &qtb_adaptive_rate)) {
        PRINT_INPUT_ERROR("adapt_rate should be a number.");
      }
      if (qtb_adaptive_rate < 0.0) {
        PRINT_INPUT_ERROR("adapt_rate should >= 0.");
      }
      i += 2;
    } else if (strcmp(params[i], "adapt_window") == 0) {
      if (i + 1 >= num_params) {
        PRINT_INPUT_ERROR("adapt_window requires a value.");
      }
      if (!is_valid_real(params[i + 1], &qtb_adaptive_window)) {
        PRINT_INPUT_ERROR("adapt_window should be a number.");
      }
      if (qtb_adaptive_window <= 0.0) {
        PRINT_INPUT_ERROR("adapt_window should > 0.");
      }
      i += 2;
    } else if (strcmp(params[i], "adapt_rate_type") == 0) {
      int type_id = 0;
      double rate_value = 0.0;
      if (i + 2 >= num_params) {
        PRINT_INPUT_ERROR("adapt_rate_type requires <type_id> <value>.");
      }
      if (!is_valid_int(params[i + 1], &type_id)) {
        PRINT_INPUT_ERROR("adapt_rate_type type_id should be an integer.");
      }
      if (type_id < 0) {
        PRINT_INPUT_ERROR("adapt_rate_type type_id should >= 0.");
      }
      if (!is_valid_real(params[i + 2], &rate_value)) {
        PRINT_INPUT_ERROR("adapt_rate_type value should be a number.");
      }
      if (rate_value <= 0.0) {
        PRINT_INPUT_ERROR("adapt_rate_type value should > 0.");
      }
      if (qtb_adaptive_rate_type_host.size() <= size_t(type_id)) {
        qtb_adaptive_rate_type_host.resize(size_t(type_id) + 1, -1.0);
      }
      qtb_adaptive_rate_type_host[type_id] = rate_value;
      i += 3;
    } else {
      PRINT_INPUT_ERROR("Unknown npt_qtb keyword.");
    }
  }

  if (t_start <= 0 || t_stop <= 0) {
    PRINT_INPUT_ERROR("npt_qtb requires temp <T_start> <T_stop>.");
  }
  if (!use_barostat) {
    PRINT_INPUT_ERROR("npt_qtb requires pressure specification (iso/aniso/tri/x/y/z).");
  }

  qtb_N_f = qtb_n_f_input;

  printf("Use NPT-QTB ensemble for this run.\n");
  printf("    Parrinello-Rahman barostat + quantum thermal bath thermostat.\n");
  printf("    QTB temperature: t_start=%g K, t_stop=%g K\n", t_start, t_stop);
  printf("    QTB tperiod=%g timesteps\n", t_period);
  printf("    QTB f_max=%g ps^-1, N_f=%d\n", qtb_f_max, qtb_N_f);
  if (qtb_use_adaptive) {
    printf("    adaptive QTB is enabled.\n");
    printf("    adapt_rate is %g.\n", qtb_adaptive_rate);
    if (qtb_adaptive_rate == 0.0) {
      printf("    adapt_rate=0 enables diagnostic-only mode (gamma spectrum will not adapt).\n");
    }
    if (qtb_adaptive_window > 0.0) {
      printf("    adapt_window is %g time_step.\n", qtb_adaptive_window);
    }
    for (int type_id = 0; type_id < int(qtb_adaptive_rate_type_host.size()); ++type_id) {
      if (qtb_adaptive_rate_type_host[type_id] > 0.0) {
        printf(
          "    adapt_rate_type for atom type %d is %g.\n",
          type_id,
          qtb_adaptive_rate_type_host[type_id]);
      }
    }
  }

  const char* sc[3][3] = {{"xx", "xy", "xz"}, {"yx", "yy", "yz"}, {"zx", "zy", "zz"}};
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      if (p_flag[a][b]) {
        printf(
          "    %s: p_start=%g, p_stop=%g, pperiod=%g\n",
          sc[a][b],
          p_start[a][b],
          p_stop[a][b],
          p_period[a][b]);
      }
    }
  }
}

Ensemble_NPT_QTB::~Ensemble_NPT_QTB(void)
{
  if (qtb_adaptive_gamma_file) {
    fclose(qtb_adaptive_gamma_file);
  }
  if (qtb_adaptive_fdt_file) {
    fclose(qtb_adaptive_fdt_file);
  }
  if (qtb_adaptive_theta_file) {
    fclose(qtb_adaptive_theta_file);
  }
  if (qtb_adaptive_fft_plan_initialized) {
    gpufftDestroy(qtb_adaptive_fft_plan);
  }
}

void Ensemble_NPT_QTB::init_mttk()
{
  Ensemble_MTTK::init_mttk();
  init_qtb();
  if (qtb_use_adaptive) {
    initialize_adaptive_qtb();
  }
}

void Ensemble_NPT_QTB::init_qtb()
{
  qtb_number_of_atoms = atom->number_of_atoms;
  qtb_number_of_atom_types = 1;
  qtb_dt = time_step;
  qtb_nfreq2 = 2 * qtb_N_f;
  qtb_time_filter_count = 1;
  qtb_adaptive_sample_count = 0;
  qtb_adaptive_update_count = 0;

  qtb_f_max_natural = qtb_f_max * TIME_UNIT_CONVERSION / 1000.0;
  int alpha_estimate = int(1.0 / (2.0 * qtb_f_max_natural * qtb_dt));
  if (alpha_estimate < 1) {
    qtb_alpha = 1;
  } else {
    qtb_alpha = alpha_estimate;
  }

  qtb_h_timestep = qtb_alpha * qtb_dt;
  qtb_fric_coef = 1.0 / (t_period * qtb_dt);
  qtb_counter_mu = 0;
  qtb_adaptive_segment_length = qtb_nfreq2;
  qtb_last_filter_temperature = -1.0;
  if (qtb_adaptive_window > 0.0) {
    int segment_estimate = int(floor(qtb_adaptive_window / qtb_alpha + 0.5));
    if (segment_estimate < 2) {
      segment_estimate = 2;
    }
    if (segment_estimate % 2 != 0) {
      segment_estimate += 1;
    }
    qtb_adaptive_segment_length = segment_estimate;
  }
  qtb_adaptive_gamma_min = 0.01 * qtb_fric_coef;
  qtb_adaptive_gamma_max = 20.0 * qtb_fric_coef;
  qtb_filter_is_dirty = true;
  qtb_adaptive_initialized = false;
  qtb_adaptive_fft_plan_initialized = false;
  qtb_adaptive_gamma_floor_warning_issued = false;
  qtb_last_theta_dump_temperature = -1.0;

  qtb_time_H_host.resize(qtb_nfreq2, 0.0);
  qtb_time_H_device.resize(qtb_nfreq2);
  qtb_gamma_spectrum_host.resize(qtb_nfreq2, qtb_fric_coef);

  const size_t history_size = size_t(qtb_number_of_atoms) * size_t(qtb_nfreq2);
  qtb_random_array_0.resize(history_size);
  qtb_random_array_1.resize(history_size);
  qtb_random_array_2.resize(history_size);
  qtb_fran.resize(size_t(qtb_number_of_atoms) * 3);

  qtb_curand_states.resize(qtb_number_of_atoms);
  initialize_curand_states<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_curand_states.data(), qtb_number_of_atoms, rand());
  GPU_CHECK_KERNEL

  gpu_initialize_qtb_history<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_curand_states.data(),
    qtb_number_of_atoms,
    qtb_nfreq2,
    qtb_random_array_0.data(),
    qtb_random_array_1.data(),
    qtb_random_array_2.data());
  GPU_CHECK_KERNEL
}

void Ensemble_NPT_QTB::initialize_adaptive_qtb()
{
  if (!qtb_use_adaptive || qtb_adaptive_initialized) {
    return;
  }

  qtb_number_of_atom_types = int(atom->cpu_type_size.size());
  if (qtb_number_of_atom_types < 1) {
    qtb_number_of_atom_types = 1;
  }
  qtb_time_filter_count = qtb_number_of_atom_types;
  qtb_atom_type_counts_host.resize(qtb_number_of_atom_types, 0);
  qtb_atom_type_masses_host.resize(qtb_number_of_atom_types, 0.0);
  for (int type = 0; type < int(atom->cpu_type_size.size()); ++type) {
    qtb_atom_type_counts_host[type] = atom->cpu_type_size[type];
  }
  for (int n = 0; n < int(atom->cpu_type.size()); ++n) {
    const int type = atom->cpu_type[n];
    if (qtb_atom_type_masses_host[type] <= 0.0) {
      qtb_atom_type_masses_host[type] = atom->cpu_mass[n];
    }
  }
  if (qtb_adaptive_rate_type_host.size() < size_t(qtb_number_of_atom_types)) {
    qtb_adaptive_rate_type_host.resize(size_t(qtb_number_of_atom_types), -1.0);
  }
  for (int type = 0; type < qtb_number_of_atom_types; ++type) {
    if (qtb_adaptive_rate_type_host[type] <= 0.0) {
      qtb_adaptive_rate_type_host[type] = qtb_adaptive_rate;
    }
  }

  qtb_time_H_host.resize(size_t(qtb_time_filter_count) * size_t(qtb_nfreq2), 0.0);
  qtb_time_H_device.resize(size_t(qtb_time_filter_count) * size_t(qtb_nfreq2));
  qtb_gamma_spectrum_host.resize(
    size_t(qtb_time_filter_count) * size_t(qtb_nfreq2), qtb_fric_coef);
  qtb_adaptive_vv_host.resize(size_t(qtb_number_of_atom_types) * size_t(qtb_nfreq2), 0.0);
  qtb_adaptive_vr_raw_host.resize(size_t(qtb_number_of_atom_types) * size_t(qtb_nfreq2), 0.0);
  qtb_adaptive_vr_host.resize(size_t(qtb_number_of_atom_types) * size_t(qtb_nfreq2), 0.0);
  qtb_adaptive_ff_host.resize(size_t(qtb_number_of_atom_types) * size_t(qtb_nfreq2), 0.0);
  qtb_adaptive_vv_segment_host.resize(
    size_t(qtb_number_of_atom_types) * size_t(qtb_adaptive_segment_length), 0.0);
  qtb_adaptive_vr_segment_raw_host.resize(
    size_t(qtb_number_of_atom_types) * size_t(qtb_adaptive_segment_length), 0.0);
  qtb_adaptive_ff_segment_host.resize(
    size_t(qtb_number_of_atom_types) * size_t(qtb_adaptive_segment_length), 0.0);
  qtb_adaptive_vv_sums.resize(size_t(qtb_number_of_atom_types) * size_t(qtb_adaptive_segment_length));
  qtb_adaptive_vr_sums.resize(size_t(qtb_number_of_atom_types) * size_t(qtb_adaptive_segment_length));
  qtb_adaptive_ff_sums.resize(size_t(qtb_number_of_atom_types) * size_t(qtb_adaptive_segment_length));
  qtb_adaptive_velocity_history.resize(
    size_t(qtb_number_of_atoms) * 3 * size_t(qtb_adaptive_segment_length));
  qtb_adaptive_random_history.resize(
    size_t(qtb_number_of_atoms) * 3 * size_t(qtb_adaptive_segment_length));

  int length[1] = {qtb_adaptive_segment_length};
  if (gpufftPlanMany(
        &qtb_adaptive_fft_plan,
        1,
        length,
        NULL,
        1,
        qtb_adaptive_segment_length,
        NULL,
        1,
        qtb_adaptive_segment_length,
        GPUFFT_C2C,
        qtb_number_of_atoms * 3) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: adaptive QTB plan creation failed" << std::endl;
    exit(1);
  }

  qtb_adaptive_fft_plan_initialized = true;
  qtb_adaptive_initialized = true;
  qtb_filter_is_dirty = true;

  qtb_adaptive_gamma_file = my_fopen("qtb_adaptive_gamma.out", "w");
  qtb_adaptive_fdt_file = my_fopen("qtb_adaptive_fdt.out", "w");
  qtb_adaptive_theta_file = my_fopen("qtb_theta_correction.out", "w");
  fprintf(qtb_adaptive_gamma_file, "# update step time_ps type frequency_ps^-1 gamma_ps^-1\n");
  fprintf(
    qtb_adaptive_fdt_file,
    "# update step time_ps type frequency_ps^-1 delta_fdt_ps^-2 c_vv c_vr c_vr_raw c_ff gamma_ps^-1\n");
  fprintf(
    qtb_adaptive_theta_file,
    "# step time_ps frequency_ps^-1 theta_raw_eV theta_corrected_eV correction_ratio\n");
}

void Ensemble_NPT_QTB::get_target_temp()
{
  t_target = t_start + (t_stop - t_start) * get_delta();
}

void Ensemble_NPT_QTB::write_adaptive_qtb_diagnostics()
{
  if (!qtb_adaptive_gamma_file || !qtb_adaptive_fdt_file) {
    return;
  }

  const double time_ps = (*current_step) * qtb_dt * TIME_UNIT_CONVERSION / 1000.0;
  const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;

  for (int type = 0; type < qtb_number_of_atom_types; ++type) {
    const int type_offset = type * qtb_nfreq2;
    const double dof_count = fmax(1.0, 3.0 * qtb_atom_type_counts_host[type]);
    const double normalization =
      1.0 / (dof_count * qtb_adaptive_segment_length * qtb_adaptive_segment_length);

    for (int k = 0; k <= qtb_N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(qtb_N_f, k);
      const double gamma_k = qtb_gamma_spectrum_host[gamma_index];
      const double frequency_ps =
        k * frequency_conversion / (qtb_nfreq2 * qtb_h_timestep);
      const double gamma_value = gamma_k * frequency_conversion;
      const double c_vv = qtb_adaptive_vv_host[type_offset + k] * normalization;
      const double c_vr = qtb_adaptive_vr_host[type_offset + k] * normalization;
      const double c_vr_raw = qtb_adaptive_vr_raw_host[type_offset + k] * normalization;
      const double c_ff = qtb_adaptive_ff_host[type_offset + k] * normalization;
      const double delta_fdt =
        (qtb_adaptive_vr_host[type_offset + k] -
         gamma_k * qtb_adaptive_vv_host[type_offset + k]) *
        normalization * frequency_conversion * frequency_conversion;

      fprintf(
        qtb_adaptive_gamma_file,
        "%d %d %.8f %d %.8f %.8e\n",
        qtb_adaptive_update_count,
        *current_step,
        time_ps,
        type,
        frequency_ps,
        gamma_value);
      fprintf(
        qtb_adaptive_fdt_file,
        "%d %d %.8f %d %.8f %.8e %.8e %.8e %.8e %.8e %.8e\n",
        qtb_adaptive_update_count,
        *current_step,
        time_ps,
        type,
        frequency_ps,
        delta_fdt,
        c_vv,
        c_vr,
        c_vr_raw,
        c_ff,
        gamma_value);
    }
    fprintf(qtb_adaptive_gamma_file, "\n");
    fprintf(qtb_adaptive_fdt_file, "\n");
  }

  fflush(qtb_adaptive_gamma_file);
  fflush(qtb_adaptive_fdt_file);
}

void Ensemble_NPT_QTB::write_theta_correction_diagnostics(
  const double target_temperature,
  const std::vector<double>& raw_theta,
  const std::vector<double>& corrected_theta)
{
  if (!qtb_adaptive_theta_file) {
    return;
  }
  if (fabs(target_temperature - qtb_last_theta_dump_temperature) < 1.0e-12) {
    return;
  }

  const double time_ps = (*current_step) * qtb_dt * TIME_UNIT_CONVERSION / 1000.0;
  const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;

  for (int k = 0; k <= qtb_N_f; ++k) {
    const double frequency_ps =
      k * frequency_conversion / (qtb_nfreq2 * qtb_h_timestep);
    double ratio = 0.0;
    if (raw_theta[k] != 0.0) {
      ratio = (corrected_theta[k] - raw_theta[k]) / raw_theta[k];
    }
    fprintf(
      qtb_adaptive_theta_file,
      "%d %.8f %.8f %.8e %.8e %.8e\n",
      *current_step,
      time_ps,
      frequency_ps,
      raw_theta[k],
      corrected_theta[k],
      ratio);
  }
  fprintf(qtb_adaptive_theta_file, "\n");
  fflush(qtb_adaptive_theta_file);
  qtb_last_theta_dump_temperature = target_temperature;
}

void Ensemble_NPT_QTB::qtb_update_time_filter(const double target_temperature)
{
  if (!qtb_filter_is_dirty &&
      fabs(target_temperature - qtb_last_filter_temperature) < 1.0e-12) {
    return;
  }

  std::vector<double> omega_H(qtb_nfreq2, 0.0);
  std::vector<double> raw_theta;
  std::vector<double> corrected_theta;
  compute_corrected_qtb_energy_density(
    qtb_N_f,
    qtb_nfreq2,
    qtb_h_timestep,
    qtb_fric_coef,
    target_temperature,
    raw_theta,
    corrected_theta);
  write_theta_correction_diagnostics(target_temperature, raw_theta, corrected_theta);

  for (int filter = 0; filter < qtb_time_filter_count; ++filter) {
    const int filter_offset = filter * qtb_nfreq2;
    for (int k = 0; k < qtb_nfreq2; ++k) {
      const double k_shift = k - qtb_N_f;
      const double gamma_k = qtb_gamma_spectrum_host[filter_offset + k];
      const int positive_index = abs(int(k_shift));
      const double omega_k = positive_index * PI / (qtb_N_f * qtb_h_timestep);
      const double g_k = get_ou_spectrum_correction(omega_k, qtb_fric_coef, qtb_h_timestep);
      const double theta_k = corrected_theta[positive_index];
      const double prefactor = 24.0 * gamma_k * theta_k * g_k / qtb_h_timestep;

      if (k == qtb_N_f) {
        omega_H[k] = sqrt(fmax(0.0, prefactor));
        continue;
      }

      omega_H[k] = sqrt(fmax(0.0, prefactor));
      const double numerator = sin(k_shift * PI / (2.0 * qtb_alpha * qtb_N_f));
      const double denominator = sin(k_shift * PI / (2.0 * qtb_N_f));
      omega_H[k] *= qtb_alpha * numerator / denominator;
    }

    for (int n = 0; n < qtb_nfreq2; ++n) {
      double value = 0.0;
      const double t_n = n - qtb_N_f;
      for (int k = 0; k < qtb_nfreq2; ++k) {
        const double omega_k = (k - qtb_N_f) * PI / qtb_N_f;
        value += omega_H[k] * cos(omega_k * t_n);
      }
      qtb_time_H_host[filter_offset + n] = value / qtb_nfreq2;
    }
  }

  qtb_time_H_device.copy_from_host(qtb_time_H_host.data());
  qtb_last_filter_temperature = target_temperature;
  qtb_filter_is_dirty = false;
}

void Ensemble_NPT_QTB::qtb_refresh_colored_random_force()
{
  gpu_refresh_qtb_random_force<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_curand_states.data(),
    qtb_number_of_atoms,
    qtb_nfreq2,
    qtb_time_filter_count,
    atom->type.data(),
    qtb_time_H_device.data(),
    atom->mass.data(),
    qtb_random_array_0.data(),
    qtb_random_array_1.data(),
    qtb_random_array_2.data(),
    qtb_fran.data(),
    qtb_fran.data() + qtb_number_of_atoms,
    qtb_fran.data() + qtb_number_of_atoms * 2);
  GPU_CHECK_KERNEL
}

void Ensemble_NPT_QTB::qtb_apply_half_step()
{
  const double dt_half = 0.5 * qtb_dt;
  const double c1 = exp(-qtb_fric_coef * dt_half);
  const double c2 =
    qtb_fric_coef > 0.0 ? (1.0 - c1) / qtb_fric_coef : dt_half;

  gpu_apply_qtb_ou_step<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_number_of_atoms,
    c1,
    c2,
    atom->mass.data(),
    qtb_fran.data(),
    qtb_fran.data() + qtb_number_of_atoms,
    qtb_fran.data() + qtb_number_of_atoms * 2,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + qtb_number_of_atoms,
    atom->velocity_per_atom.data() + qtb_number_of_atoms * 2);
  GPU_CHECK_KERNEL

  gpu_find_momentum<<<4, 1024>>>(
    qtb_number_of_atoms,
    atom->mass.data(),
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + qtb_number_of_atoms,
    atom->velocity_per_atom.data() + qtb_number_of_atoms * 2);
  GPU_CHECK_KERNEL

  gpu_correct_momentum<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_number_of_atoms,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + qtb_number_of_atoms,
    atom->velocity_per_atom.data() + qtb_number_of_atoms * 2);
  GPU_CHECK_KERNEL
}

void Ensemble_NPT_QTB::sample_adaptive_qtb()
{
  if (!qtb_use_adaptive || !qtb_adaptive_initialized) {
    return;
  }

  gpu_store_qtb_sample<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_number_of_atoms,
    qtb_adaptive_sample_count,
    qtb_adaptive_segment_length,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + qtb_number_of_atoms,
    atom->velocity_per_atom.data() + 2 * qtb_number_of_atoms,
    qtb_fran.data(),
    qtb_fran.data() + qtb_number_of_atoms,
    qtb_fran.data() + 2 * qtb_number_of_atoms,
    qtb_adaptive_velocity_history.data(),
    qtb_adaptive_random_history.data());
  GPU_CHECK_KERNEL

  qtb_adaptive_sample_count++;
  if (qtb_adaptive_sample_count == qtb_adaptive_segment_length) {
    adapt_random_force_spectrum();
    qtb_adaptive_sample_count = 0;
  }
}

void Ensemble_NPT_QTB::adapt_random_force_spectrum()
{
  const int num_sums = qtb_number_of_atom_types * qtb_adaptive_segment_length;
  int clamped_gamma_bins = 0;

  gpu_zero_qtb_spectrum_sums<<<(num_sums - 1) / 128 + 1, 128>>>(
    num_sums,
    qtb_adaptive_vv_sums.data(),
    qtb_adaptive_vr_sums.data(),
    qtb_adaptive_ff_sums.data());
  GPU_CHECK_KERNEL

  if (gpufftExecC2C(
        qtb_adaptive_fft_plan,
        qtb_adaptive_velocity_history.data(),
        qtb_adaptive_velocity_history.data(),
        GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: adaptive QTB velocity transform failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(
        qtb_adaptive_fft_plan,
        qtb_adaptive_random_history.data(),
        qtb_adaptive_random_history.data(),
        GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: adaptive QTB random-force transform failed" << std::endl;
    exit(1);
  }

  gpu_accumulate_qtb_spectra<<<(qtb_number_of_atoms * 3 - 1) / 128 + 1, 128>>>(
    qtb_number_of_atoms,
    qtb_adaptive_segment_length,
    atom->type.data(),
    atom->mass.data(),
    qtb_adaptive_velocity_history.data(),
    qtb_adaptive_random_history.data(),
    qtb_adaptive_vv_sums.data(),
    qtb_adaptive_vr_sums.data(),
    qtb_adaptive_ff_sums.data());
  GPU_CHECK_KERNEL

  qtb_adaptive_vv_sums.copy_to_host(qtb_adaptive_vv_segment_host.data());
  qtb_adaptive_vr_sums.copy_to_host(qtb_adaptive_vr_segment_raw_host.data());
  qtb_adaptive_ff_sums.copy_to_host(qtb_adaptive_ff_segment_host.data());

  for (int type = 0; type < qtb_number_of_atom_types; ++type) {
    const int type_offset = type * qtb_nfreq2;
    const int segment_type_offset = type * qtb_adaptive_segment_length;
    const double atom_mass = qtb_atom_type_masses_host[type];
    // The random force is refreshed once per coarse h_timestep block and kept
    // constant within that block. The sampled velocity therefore differs from
    // the block-centered velocity by half a coarse block of OU propagation.
    const double half_decay = exp(-0.5 * qtb_fric_coef * qtb_h_timestep);
    const double half_response =
      qtb_fric_coef > 0.0 ? (1.0 - half_decay) / qtb_fric_coef : 0.5 * qtb_h_timestep;
    std::vector<double> delta_fdt(size_t(qtb_N_f) + 1, 0.0);
    double delta_norm_sq = 0.0;

    for (int k = 0; k <= qtb_N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(qtb_N_f, k);
      const double gamma_k = qtb_gamma_spectrum_host[gamma_index];
      const double adaptive_bin = double(k) * qtb_adaptive_segment_length / qtb_nfreq2;
      qtb_adaptive_vv_host[type_offset + k] = interpolate_adaptive_spectrum(
        qtb_adaptive_vv_segment_host,
        qtb_adaptive_segment_length,
        segment_type_offset,
        adaptive_bin);
      qtb_adaptive_vr_raw_host[type_offset + k] = interpolate_adaptive_spectrum(
        qtb_adaptive_vr_segment_raw_host,
        qtb_adaptive_segment_length,
        segment_type_offset,
        adaptive_bin);
      qtb_adaptive_ff_host[type_offset + k] = interpolate_adaptive_spectrum(
        qtb_adaptive_ff_segment_host,
        qtb_adaptive_segment_length,
        segment_type_offset,
        adaptive_bin);
      qtb_adaptive_vr_host[type_offset + k] =
        (qtb_adaptive_vr_raw_host[type_offset + k] -
         half_response * qtb_adaptive_ff_host[type_offset + k] / atom_mass) /
        half_decay;
      const double delta_fdt_k =
        qtb_adaptive_vr_host[type_offset + k] -
        gamma_k * qtb_adaptive_vv_host[type_offset + k];
      delta_fdt[k] = delta_fdt_k;
      delta_norm_sq += delta_fdt_k * delta_fdt_k;
    }

    const double delta_norm = sqrt(delta_norm_sq);
    if (delta_norm < 1.0e-30) {
      continue;
    }

    for (int k = 0; k <= qtb_N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(qtb_N_f, k);
      const double adaptive_rate_type = qtb_adaptive_rate_type_host[type];
      const double gamma_old = qtb_gamma_spectrum_host[gamma_index];
      const double gamma_new =
        gamma_old + adaptive_rate_type * gamma_old * delta_fdt[k] / delta_norm;
      const double gamma_clamped =
        fmax(qtb_adaptive_gamma_min, fmin(qtb_adaptive_gamma_max, gamma_new));
      if (gamma_clamped <= qtb_adaptive_gamma_min && gamma_new < qtb_adaptive_gamma_min) {
        clamped_gamma_bins++;
      }
      qtb_gamma_spectrum_host[gamma_index] = gamma_clamped;

      if (k > 0 && k < qtb_N_f) {
        qtb_gamma_spectrum_host[type_offset + qtb_N_f - k] = gamma_clamped;
      }
    }
  }

  qtb_adaptive_update_count++;
  if (clamped_gamma_bins > 0 && !qtb_adaptive_gamma_floor_warning_issued) {
    const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;
    std::cout << "Warning: adQTB clamped " << clamped_gamma_bins
              << " gamma bins to the minimum floor ("
              << qtb_adaptive_gamma_min * frequency_conversion
              << " ps^-1). Consider increasing the global friction if this persists."
              << std::endl;
    qtb_adaptive_gamma_floor_warning_issued = true;
  }
  write_adaptive_qtb_diagnostics();
  qtb_filter_is_dirty = true;
}

void Ensemble_NPT_QTB::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0) {
    init_mttk();
  }

  nhc_press_integrate();

  get_target_temp();
  if (qtb_counter_mu == 0) {
    qtb_update_time_filter(t_target);
    qtb_refresh_colored_random_force();
  }
  qtb_apply_half_step();

  get_h_matrix_from_box();
  get_target_pressure();
  nh_omega_dot();
  nh_v_press();

  velocity_verlet_v();
  propagate_box();
  velocity_verlet_x();
  propagate_box();
}

void Ensemble_NPT_QTB::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet_v();

  get_h_matrix_from_box();
  nh_v_press();
  nh_omega_dot();

  qtb_apply_half_step();
  if (qtb_counter_mu == qtb_alpha - 1) {
    sample_adaptive_qtb();
  }

  nhc_press_integrate();
  find_thermo();

  qtb_counter_mu = (qtb_counter_mu + 1) % qtb_alpha;
}
