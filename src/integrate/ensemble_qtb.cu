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
The QTB thermostat based on a colored noise filter:
[1] Dammak, T., et al. Phys. Rev. Lett. 103, 190601 (2009).
------------------------------------------------------------------------------*/

#include "ensemble_qtb.cuh"
#include "langevin_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace
{
static __device__ double draw_qtb_white_noise(gpurandState* state)
{
  return gpurand_normal_double(state) / sqrt(12.0);
}

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

static void smooth_positive_spectrum(
  std::vector<double>& spectrum,
  const int offset,
  const int positive_size,
  const double width_bins)
{
  if (width_bins <= 0.0 || positive_size <= 1) {
    return;
  }

  std::vector<double> input(size_t(positive_size), 0.0);
  for (int i = 0; i < positive_size; ++i) {
    input[i] = spectrum[offset + i];
  }

  for (int i = 0; i < positive_size; ++i) {
    const int first = std::max(0, int(floor(i - width_bins)));
    const int last = std::min(positive_size - 1, int(ceil(i + width_bins)));
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (int j = first; j <= last; ++j) {
      const double x = (j - i) / width_bins;
      if (fabs(x) >= 1.0) {
        continue;
      }
      const double weight = exp(-1.0 / (1.0 + x)) * exp(-1.0 / (1.0 - x));
      weighted_sum += input[j] * weight;
      weight_sum += weight;
    }
    if (weight_sum > 0.0) {
      spectrum[offset + i] = weighted_sum / weight_sum;
    }
  }
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
      const double kernel_value =
        fric_coef * omega_i_sq * weight / (PI * denom);
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
    // The deconvolution becomes ill-conditioned in the first few bins near
    // omega = 0, where the corrected spectrum should smoothly recover the
    // classical limit. Blend those bins back to a stable low-frequency anchor
    // over approximately one friction linewidth.
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
      random_array_0[offset + m] = draw_qtb_white_noise(&state);
      random_array_1[offset + m] = draw_qtb_white_noise(&state);
      random_array_2[offset + m] = draw_qtb_white_noise(&state);
    }
    states[n] = state;
  }
}

static __global__ void gpu_refresh_qtb_random_force(
  gpurandState* states,
  const int N,
  const int nfreq2,
  const bool shift_history,
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

    if (shift_history) {
      for (int m = 0; m < nfreq2 - 1; ++m) {
        random_array_0[offset + m] = random_array_0[offset + m + 1];
        random_array_1[offset + m] = random_array_1[offset + m + 1];
        random_array_2[offset + m] = random_array_2[offset + m + 1];
      }
      random_array_0[offset + nfreq2 - 1] = draw_qtb_white_noise(&state);
      random_array_1[offset + nfreq2 - 1] = draw_qtb_white_noise(&state);
      random_array_2[offset + nfreq2 - 1] = draw_qtb_white_noise(&state);
    }

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
    const double scale = sqrt_mass;
    fran_x[n] = sum_x * scale;
    fran_y[n] = sum_y * scale;
    fran_z[n] = sum_z * scale;

    states[n] = state;
  }
}

static __global__ void gpu_qtb_position_half_step(
  const int N,
  const int fixed_group,
  const int move_group,
  const double move_velocity_x,
  const double move_velocity_y,
  const double move_velocity_z,
  const int* group_id,
  const double dt_half,
  const double* vx,
  const double* vy,
  const double* vz,
  double* x,
  double* y,
  double* z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    if (group_id[n] == fixed_group) {
      return;
    }
    if (group_id[n] == move_group) {
      x[n] += move_velocity_x * dt_half;
      y[n] += move_velocity_y * dt_half;
      z[n] += move_velocity_z * dt_half;
      return;
    }
    x[n] += vx[n] * dt_half;
    y[n] += vy[n] * dt_half;
    z[n] += vz[n] * dt_half;
  }
}

static __global__ void gpu_qtb_position_half_step(
  const int N,
  const double dt_half,
  const double* vx,
  const double* vy,
  const double* vz,
  double* x,
  double* y,
  double* z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    x[n] += vx[n] * dt_half;
    y[n] += vy[n] * dt_half;
    z[n] += vz[n] * dt_half;
  }
}

static __global__ void gpu_qtb_force_half_step(
  const int N,
  const int fixed_group,
  const int move_group,
  const int* group_id,
  const double dt_half,
  const double* mass,
  const double* fx,
  const double* fy,
  const double* fz,
  double* vx,
  double* vy,
  double* vz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    if (group_id[n] == fixed_group || group_id[n] == move_group) {
      vx[n] = 0.0;
      vy[n] = 0.0;
      vz[n] = 0.0;
      return;
    }
    const double mass_inv = 1.0 / mass[n];
    vx[n] += dt_half * fx[n] * mass_inv;
    vy[n] += dt_half * fy[n] * mass_inv;
    vz[n] += dt_half * fz[n] * mass_inv;
  }
}

static __global__ void gpu_qtb_force_half_step(
  const int N,
  const double dt_half,
  const double* mass,
  const double* fx,
  const double* fy,
  const double* fz,
  double* vx,
  double* vy,
  double* vz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double mass_inv = 1.0 / mass[n];
    vx[n] += dt_half * fx[n] * mass_inv;
    vy[n] += dt_half * fy[n] * mass_inv;
    vz[n] += dt_half * fz[n] * mass_inv;
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

static __global__ void gpu_store_qtb_ou_center_sample(
  const int N,
  const int sample_index,
  const int segment_length,
  const double c1_sqrt,
  const double dt_half,
  const double* mass,
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
    const double mass_inv = 1.0 / mass[n];

    velocity_history[offset_x].x = float(c1_sqrt * vx[n] + dt_half * fran_x[n] * mass_inv);
    velocity_history[offset_x].y = 0.0f;
    velocity_history[offset_y].x = float(c1_sqrt * vy[n] + dt_half * fran_y[n] * mass_inv);
    velocity_history[offset_y].y = 0.0f;
    velocity_history[offset_z].x = float(c1_sqrt * vz[n] + dt_half * fran_z[n] * mass_inv);
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

void Ensemble_QTB::init_qtb_common(
  int N,
  double T,
  double Tc,
  double dt_input,
  double f_max_input,
  int N_f_input,
  bool use_adaptive_qtb_input,
  bool use_theta_correction_input,
  double adaptive_rate_input,
  double adaptive_tau_average_input,
  double adaptive_tau_adapt_input,
  double adaptive_smooth_width_input,
  double adaptive_window_input,
  const std::vector<double>& adaptive_rate_type_input)
{
  number_of_atoms = N;
  number_of_atom_types = 1;
  temperature = T;
  temperature_coupling = Tc;
  dt = dt_input;
  N_f = N_f_input;
  nfreq2 = 2 * N_f;
  time_filter_count = 1;
  adaptive_sample_count = 0;
  adaptive_update_count = 0;

  f_max_natural = f_max_input * TIME_UNIT_CONVERSION / 1000.0;

  int alpha_estimate = int(1.0 / (2.0 * f_max_natural * dt));
  if (alpha_estimate < 1) {
    alpha = 1;
  } else {
    alpha = alpha_estimate;
  }

  h_timestep = alpha * dt;
  fric_coef = 1.0 / (temperature_coupling * dt);
  counter_mu = 0;
  adaptive_segment_length = nfreq2;
  last_filter_temperature = -1.0;
  adaptive_rate = adaptive_rate_input;
  adaptive_window = adaptive_window_input;
  if (adaptive_window_input > 0.0) {
    int segment_estimate = int(floor(adaptive_window_input + 0.5));
    if (segment_estimate < 2) {
      segment_estimate = 2;
    }
    if (segment_estimate % 2 != 0) {
      segment_estimate += 1;
    }
    adaptive_segment_length = segment_estimate;
  }
  adaptive_tau_average =
    adaptive_tau_average_input > 0.0
      ? adaptive_tau_average_input * dt
      : 10.0 * adaptive_segment_length * dt;
  adaptive_tau_adapt = adaptive_tau_adapt_input > 0.0 ? adaptive_tau_adapt_input * dt : 0.0;
  adaptive_smooth_width = adaptive_smooth_width_input;
  adaptive_gamma_min = 0.01 * fric_coef;
  adaptive_gamma_max = 20.0 * fric_coef;
  use_adaptive_qtb = use_adaptive_qtb_input;
  use_theta_correction = use_theta_correction_input;
  filter_is_dirty = true;
  adaptive_qtb_initialized = false;
  adaptive_fft_plan_initialized = false;
  adaptive_gamma_floor_warning_issued = false;
  adaptive_gamma_file = nullptr;
  adaptive_fdt_file = nullptr;
  adaptive_theta_file = nullptr;
  last_theta_dump_temperature = -1.0;
  adaptive_rate_type_host = adaptive_rate_type_input;

  time_H_host.resize(nfreq2, 0.0);
  time_H_device.resize(nfreq2);
  gamma_spectrum_host.resize(nfreq2, fric_coef);
  gamma_initial_spectrum_host = gamma_spectrum_host;

  random_array_0.resize(size_t(number_of_atoms) * size_t(nfreq2));
  random_array_1.resize(size_t(number_of_atoms) * size_t(nfreq2));
  random_array_2.resize(size_t(number_of_atoms) * size_t(nfreq2));
  fran.resize(size_t(number_of_atoms) * 3);

  curand_states.resize(number_of_atoms);
  initialize_curand_states<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(), number_of_atoms, rand());
  GPU_CHECK_KERNEL

  gpu_initialize_qtb_history<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(),
    number_of_atoms,
    nfreq2,
    random_array_0.data(),
    random_array_1.data(),
    random_array_2.data());
  GPU_CHECK_KERNEL
}

// NVT-QTB constructor
Ensemble_QTB::Ensemble_QTB(
  int t,
  int N,
  double T,
  double Tc,
  double dt_input,
  double f_max,
  int N_f_input,
  bool use_adaptive_qtb_input,
  bool use_theta_correction_input,
  double adaptive_rate_input,
  double adaptive_tau_average_input,
  double adaptive_tau_adapt_input,
  double adaptive_smooth_width_input,
  double adaptive_window_input,
  const std::vector<double>& adaptive_rate_type_input)
{
  type = t;
  num_target_pressure_components = 0;
  init_qtb_common(
    N,
    T,
    Tc,
    dt_input,
    f_max,
    N_f_input,
    use_adaptive_qtb_input,
    use_theta_correction_input,
    adaptive_rate_input,
    adaptive_tau_average_input,
    adaptive_tau_adapt_input,
    adaptive_smooth_width_input,
    adaptive_window_input,
    adaptive_rate_type_input);
}

Ensemble_QTB::~Ensemble_QTB(void)
{
  if (adaptive_qtb_initialized) {
    write_adaptive_gamma_restart();
  }
  if (adaptive_gamma_file) {
    fclose(adaptive_gamma_file);
  }
  if (adaptive_fdt_file) {
    fclose(adaptive_fdt_file);
  }
  if (adaptive_theta_file) {
    fclose(adaptive_theta_file);
  }
  if (adaptive_fft_plan_initialized) {
    gpufftDestroy(adaptive_fft_plan);
  }
}

void Ensemble_QTB::initialize_adaptive_qtb()
{
  if (!use_adaptive_qtb || adaptive_qtb_initialized) {
    return;
  }

  number_of_atom_types = int(atom->cpu_type_size.size());
  if (number_of_atom_types < 1) {
    number_of_atom_types = 1;
  }
  time_filter_count = number_of_atom_types;
  atom_type_counts_host.resize(number_of_atom_types, 0);
  atom_type_masses_host.resize(number_of_atom_types, 0.0);
  for (int type = 0; type < int(atom->cpu_type_size.size()); ++type) {
    atom_type_counts_host[type] = atom->cpu_type_size[type];
  }
  for (int n = 0; n < int(atom->cpu_type.size()); ++n) {
    const int type = atom->cpu_type[n];
    if (atom_type_masses_host[type] <= 0.0) {
      atom_type_masses_host[type] = atom->cpu_mass[n];
    }
  }
  if (adaptive_rate_type_host.size() < size_t(number_of_atom_types)) {
    adaptive_rate_type_host.resize(size_t(number_of_atom_types), -1.0);
  }
  for (int type = 0; type < number_of_atom_types; ++type) {
    if (adaptive_rate_type_host[type] <= 0.0) {
      adaptive_rate_type_host[type] = adaptive_rate;
    }
  }

  time_H_host.resize(size_t(time_filter_count) * size_t(nfreq2), 0.0);
  time_H_device.resize(size_t(time_filter_count) * size_t(nfreq2));
  gamma_spectrum_host.resize(size_t(time_filter_count) * size_t(nfreq2), fric_coef);
  gamma_initial_spectrum_host = gamma_spectrum_host;
  adaptive_vv_host.resize(size_t(number_of_atom_types) * size_t(nfreq2), 0.0);
  adaptive_vr_raw_host.resize(size_t(number_of_atom_types) * size_t(nfreq2), 0.0);
  adaptive_vr_host.resize(size_t(number_of_atom_types) * size_t(nfreq2), 0.0);
  adaptive_ff_host.resize(size_t(number_of_atom_types) * size_t(nfreq2), 0.0);
  adaptive_vv_average_host.resize(size_t(number_of_atom_types) * size_t(nfreq2), 0.0);
  adaptive_vr_average_host.resize(size_t(number_of_atom_types) * size_t(nfreq2), 0.0);
  adaptive_gamma_running_host = gamma_spectrum_host;
  adaptive_average_count_host.resize(size_t(number_of_atom_types), 0);
  adaptive_gamma_count_host.resize(size_t(number_of_atom_types), 0);
  adaptive_vv_segment_host.resize(size_t(number_of_atom_types) * size_t(adaptive_segment_length), 0.0);
  adaptive_vr_segment_raw_host.resize(
    size_t(number_of_atom_types) * size_t(adaptive_segment_length), 0.0);
  adaptive_ff_segment_host.resize(size_t(number_of_atom_types) * size_t(adaptive_segment_length), 0.0);
  adaptive_vv_sums.resize(size_t(number_of_atom_types) * size_t(adaptive_segment_length));
  adaptive_vr_sums.resize(size_t(number_of_atom_types) * size_t(adaptive_segment_length));
  adaptive_ff_sums.resize(size_t(number_of_atom_types) * size_t(adaptive_segment_length));
  adaptive_velocity_history.resize(size_t(number_of_atoms) * 3 * size_t(adaptive_segment_length));
  adaptive_random_history.resize(size_t(number_of_atoms) * 3 * size_t(adaptive_segment_length));

  int length[1] = {adaptive_segment_length};
  if (gpufftPlanMany(
        &adaptive_fft_plan,
        1,
        length,
        NULL,
        1,
        adaptive_segment_length,
        NULL,
        1,
        adaptive_segment_length,
        GPUFFT_C2C,
        number_of_atoms * 3) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: adaptive QTB plan creation failed" << std::endl;
    exit(1);
  }

  adaptive_fft_plan_initialized = true;
  adaptive_qtb_initialized = true;
  filter_is_dirty = true;

  if (load_adaptive_gamma_restart()) {
    adaptive_gamma_running_host = gamma_spectrum_host;
    gamma_initial_spectrum_host = gamma_spectrum_host;
  }

  const char* diagnostic_mode = *current_step_absolute > 0 ? "a" : "w";
  adaptive_gamma_file = my_fopen("qtb_adaptive_gamma.out", diagnostic_mode);
  adaptive_fdt_file = my_fopen("qtb_adaptive_fdt.out", diagnostic_mode);
  adaptive_theta_file = my_fopen("qtb_theta_correction.out", diagnostic_mode);
  fprintf(
    adaptive_gamma_file,
    "# update step time_ps type frequency_ps^-1 gamma_ps^-1\n");
  fprintf(
    adaptive_fdt_file,
    "# update step time_ps type frequency_ps^-1 dFDT_ps^-2 mCvv Cvf Cvf_raw Cff gamma_ps^-1\n");
  fprintf(
    adaptive_theta_file,
    "# step time_ps frequency_ps^-1 theta_raw_eV theta_corrected_eV correction_ratio\n");
}

bool Ensemble_QTB::load_adaptive_gamma_restart()
{
  std::ifstream input("qtb_gamma_restart.out");
  if (!input.good()) {
    return false;
  }

  const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;
  const double frequency_spacing = frequency_conversion / (nfreq2 * h_timestep);
  std::string line;
  int loaded = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream stream(line);
    int type = 0;
    double frequency_ps = 0.0;
    double gamma_ps = 0.0;
    if (!(stream >> type >> frequency_ps >> gamma_ps)) {
      continue;
    }
    if (type < 0 || type >= time_filter_count || frequency_spacing <= 0.0) {
      continue;
    }

    int k = int(floor(frequency_ps / frequency_spacing + 0.5));
    if (k < 0 || k > N_f) {
      continue;
    }
    const int type_offset = type * nfreq2;
    const int gamma_index = type_offset + get_gamma_index_from_fft_bin(N_f, k);
    const double gamma_value = gamma_ps / frequency_conversion;
    gamma_spectrum_host[gamma_index] = gamma_value;
    if (k > 0 && k < N_f) {
      gamma_spectrum_host[type_offset + N_f - k] = gamma_value;
    }
    loaded++;
  }

  if (loaded > 0) {
    std::cout << "Loaded " << loaded
              << " adQTB gamma values from qtb_gamma_restart.out." << std::endl;
  }
  return loaded > 0;
}

void Ensemble_QTB::write_adaptive_gamma_restart()
{
  if (gamma_spectrum_host.empty() || time_filter_count <= 0) {
    return;
  }

  std::ofstream output("qtb_gamma_restart.out");
  if (!output.good()) {
    return;
  }

  const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;
  output << "# GPUMD adQTB gamma restart\n";
  output << "# columns: type frequency_ps^-1 gamma_ps^-1\n";
  output << "# N_f " << N_f << " nfreq2 " << nfreq2
         << " filters " << time_filter_count << "\n";
  for (int type = 0; type < time_filter_count; ++type) {
    const int type_offset = type * nfreq2;
    for (int k = 0; k <= N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(N_f, k);
      const double frequency_ps = k * frequency_conversion / (nfreq2 * h_timestep);
      output << type << " " << frequency_ps << " "
             << gamma_spectrum_host[gamma_index] * frequency_conversion << "\n";
    }
    output << "\n";
  }
}

void Ensemble_QTB::write_adaptive_qtb_diagnostics()
{
  const double time_ps = (*current_step_absolute) * dt * TIME_UNIT_CONVERSION / 1000.0;
  const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;

  for (int type = 0; type < number_of_atom_types; ++type) {
    const int type_offset = type * nfreq2;
    const double dof_count = fmax(1.0, 3.0 * atom_type_counts_host[type]);
    const double normalization =
      1.0 / (dof_count * adaptive_segment_length * adaptive_segment_length);

    for (int k = 0; k <= N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(N_f, k);
      const double gamma_k = gamma_spectrum_host[gamma_index];
      const double frequency_ps = k * frequency_conversion / (nfreq2 * h_timestep);
      const double gamma_value = gamma_k * frequency_conversion;
      const double c_vv = adaptive_vv_host[type_offset + k] * normalization;
      const double c_vr = adaptive_vr_host[type_offset + k] * normalization;
      const double c_vr_raw = adaptive_vr_raw_host[type_offset + k] * normalization;
      const double c_ff = adaptive_ff_host[type_offset + k] * normalization;
      const double delta_fdt =
        (gamma_k * adaptive_vv_host[type_offset + k] -
         adaptive_vr_host[type_offset + k]) *
        normalization * frequency_conversion * frequency_conversion;

      fprintf(
        adaptive_gamma_file,
        "%d %d %.8f %d %.8f %.8e\n",
        adaptive_update_count,
        *current_step_absolute,
        time_ps,
        type,
        frequency_ps,
        gamma_value);
      fprintf(
        adaptive_fdt_file,
        "%d %d %.8f %d %.8f %.8e %.8e %.8e %.8e %.8e %.8e\n",
        adaptive_update_count,
        *current_step_absolute,
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
    fprintf(adaptive_gamma_file, "\n");
    fprintf(adaptive_fdt_file, "\n");
  }

  fflush(adaptive_gamma_file);
  fflush(adaptive_fdt_file);
}

void Ensemble_QTB::write_theta_correction_diagnostics(
  const double target_temperature,
  const std::vector<double>& raw_theta,
  const std::vector<double>& corrected_theta)
{
  if (!adaptive_theta_file) {
    return;
  }
  if (fabs(target_temperature - last_theta_dump_temperature) < 1.0e-12) {
    return;
  }

  const double time_ps = (*current_step_absolute) * dt * TIME_UNIT_CONVERSION / 1000.0;
  const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;

  for (int k = 0; k <= N_f; ++k) {
    const double frequency_ps = k * frequency_conversion / (nfreq2 * h_timestep);
    double ratio = 0.0;
    if (raw_theta[k] != 0.0) {
      ratio = (corrected_theta[k] - raw_theta[k]) / raw_theta[k];
    }
    fprintf(
      adaptive_theta_file,
      "%d %.8f %.8f %.8e %.8e %.8e\n",
      *current_step_absolute,
      time_ps,
      frequency_ps,
      raw_theta[k],
      corrected_theta[k],
      ratio);
  }
  fprintf(adaptive_theta_file, "\n");
  fflush(adaptive_theta_file);
  last_theta_dump_temperature = target_temperature;
}

void Ensemble_QTB::update_time_filter(const double target_temperature)
{
  if (!filter_is_dirty && fabs(target_temperature - last_filter_temperature) < 1.0e-12) {
    return;
  }

  std::vector<double> omega_H(nfreq2, 0.0);
  std::vector<double> raw_theta(size_t(N_f) + 1, 0.0);
  std::vector<double> corrected_theta(size_t(N_f) + 1, 0.0);

  if (use_theta_correction) {
    compute_corrected_qtb_energy_density(
      N_f, nfreq2, h_timestep, fric_coef, target_temperature, raw_theta, corrected_theta);
  } else {
    for (int k = 0; k <= N_f; ++k) {
      const double frequency = k / (nfreq2 * h_timestep);
      raw_theta[k] = get_qtb_energy_density(frequency, target_temperature);
      corrected_theta[k] = raw_theta[k];
    }
  }
  write_theta_correction_diagnostics(target_temperature, raw_theta, corrected_theta);

  for (int filter = 0; filter < time_filter_count; ++filter) {
    const int filter_offset = filter * nfreq2;
    for (int k = 0; k < nfreq2; ++k) {
      const double k_shift = k - N_f;
      const double gamma_k = gamma_spectrum_host[filter_offset + k];
      const int positive_index = abs(k_shift);
      const double omega_abs = positive_index * PI / (N_f * h_timestep);
      const double theta_k =
        use_theta_correction ? corrected_theta[positive_index] : raw_theta[positive_index];
      const double g_k = get_ou_spectrum_correction(omega_abs, fric_coef, h_timestep);
      const double prefactor = 24.0 * gamma_k * theta_k * g_k / h_timestep;
      double omega_h_k = sqrt(fmax(0.0, prefactor));

      if (k == N_f) {
        omega_H[k] = omega_h_k;
        continue;
      }

      omega_H[k] = omega_h_k;
      const double numerator = sin(k_shift * PI / (2.0 * alpha * N_f));
      const double denominator = sin(k_shift * PI / (2.0 * N_f));
      omega_H[k] *= alpha * numerator / denominator;
    }

    for (int n = 0; n < nfreq2; ++n) {
      double value = 0.0;
      const double t_n = n - N_f;
      for (int k = 0; k < nfreq2; ++k) {
        const double omega_k = (k - N_f) * PI / N_f;
        value += omega_H[k] * cos(omega_k * t_n);
      }
      time_H_host[filter_offset + n] = value / nfreq2;
    }
  }

  time_H_device.copy_from_host(time_H_host.data());
  last_filter_temperature = target_temperature;
  filter_is_dirty = false;
}

void Ensemble_QTB::sample_adaptive_qtb()
{
  if (!use_adaptive_qtb || !adaptive_qtb_initialized) {
    return;
  }

  gpu_store_qtb_sample<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    adaptive_sample_count,
    adaptive_segment_length,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms,
    fran.data(),
    fran.data() + number_of_atoms,
    fran.data() + 2 * number_of_atoms,
    adaptive_velocity_history.data(),
    adaptive_random_history.data());
  GPU_CHECK_KERNEL

  adaptive_sample_count++;
  if (adaptive_sample_count == adaptive_segment_length) {
    adapt_random_force_spectrum();
    adaptive_sample_count = 0;
  }
}

void Ensemble_QTB::sample_adaptive_qtb_ou_center()
{
  if (!use_adaptive_qtb || !adaptive_qtb_initialized) {
    return;
  }

  const double c1_sqrt = exp(-0.5 * fric_coef * dt);
  const double dt_half = 0.5 * dt;
  gpu_store_qtb_ou_center_sample<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    adaptive_sample_count,
    adaptive_segment_length,
    c1_sqrt,
    dt_half,
    atom->mass.data(),
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms,
    fran.data(),
    fran.data() + number_of_atoms,
    fran.data() + 2 * number_of_atoms,
    adaptive_velocity_history.data(),
    adaptive_random_history.data());
  GPU_CHECK_KERNEL

  adaptive_sample_count++;
  if (adaptive_sample_count == adaptive_segment_length) {
    adapt_random_force_spectrum();
    adaptive_sample_count = 0;
  }
}

void Ensemble_QTB::adapt_random_force_spectrum()
{
  const int num_sums = number_of_atom_types * adaptive_segment_length;
  int clamped_gamma_bins = 0;
  bool gamma_changed = false;
  gpu_zero_qtb_spectrum_sums<<<(num_sums - 1) / 128 + 1, 128>>>(
    num_sums, adaptive_vv_sums.data(), adaptive_vr_sums.data(), adaptive_ff_sums.data());
  GPU_CHECK_KERNEL

  if (gpufftExecC2C(
        adaptive_fft_plan,
        adaptive_velocity_history.data(),
        adaptive_velocity_history.data(),
        GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: adaptive QTB velocity transform failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(
        adaptive_fft_plan,
        adaptive_random_history.data(),
        adaptive_random_history.data(),
        GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: adaptive QTB random-force transform failed" << std::endl;
    exit(1);
  }

  gpu_accumulate_qtb_spectra<<<(number_of_atoms * 3 - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    adaptive_segment_length,
    atom->type.data(),
    atom->mass.data(),
    adaptive_velocity_history.data(),
    adaptive_random_history.data(),
    adaptive_vv_sums.data(),
    adaptive_vr_sums.data(),
    adaptive_ff_sums.data());
  GPU_CHECK_KERNEL

  adaptive_vv_sums.copy_to_host(adaptive_vv_segment_host.data());
  adaptive_vr_sums.copy_to_host(adaptive_vr_segment_raw_host.data());
  adaptive_ff_sums.copy_to_host(adaptive_ff_segment_host.data());

  for (int type = 0; type < number_of_atom_types; ++type) {
    const int type_offset = type * nfreq2;
    const int segment_type_offset = type * adaptive_segment_length;
    const double sample_timestep = dt;
    std::vector<double> delta_fdt(size_t(N_f) + 1, 0.0);
    double delta_norm_sq = 0.0;
    const double segment_time = adaptive_segment_length * sample_timestep;

    for (int k = 0; k <= N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(N_f, k);
      const double adaptive_bin =
        double(k) * adaptive_segment_length * sample_timestep / (double(nfreq2) * h_timestep);
      adaptive_vv_host[type_offset + k] = interpolate_adaptive_spectrum(
        adaptive_vv_segment_host, adaptive_segment_length, segment_type_offset, adaptive_bin);
      adaptive_vr_raw_host[type_offset + k] = interpolate_adaptive_spectrum(
        adaptive_vr_segment_raw_host, adaptive_segment_length, segment_type_offset, adaptive_bin);
      adaptive_ff_host[type_offset + k] = interpolate_adaptive_spectrum(
        adaptive_ff_segment_host, adaptive_segment_length, segment_type_offset, adaptive_bin);
      adaptive_vr_host[type_offset + k] = adaptive_vr_raw_host[type_offset + k];
    }

    if (adaptive_smooth_width > 0.0) {
      const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;
      const double frequency_spacing = frequency_conversion / (nfreq2 * h_timestep);
      const double smooth_width_bins =
        frequency_spacing > 0.0 ? adaptive_smooth_width / frequency_spacing : 0.0;
      smooth_positive_spectrum(adaptive_vv_host, type_offset, N_f + 1, smooth_width_bins);
      smooth_positive_spectrum(adaptive_vr_host, type_offset, N_f + 1, smooth_width_bins);
    }

    double average_bias_correction = 1.0;
    double elapsed_average_time = segment_time;
    const double average_decay =
      adaptive_tau_average > 0.0 ? exp(-segment_time / adaptive_tau_average) : 0.0;
    const int average_count = ++adaptive_average_count_host[type];
    elapsed_average_time = average_count * segment_time;
    average_bias_correction =
      adaptive_tau_average > 0.0 ? 1.0 - pow(average_decay, average_count) : 1.0;
    if (average_bias_correction < 1.0e-12) {
      average_bias_correction = 1.0;
    }
    for (int k = 0; k <= N_f; ++k) {
      const int index = type_offset + k;
      adaptive_vv_average_host[index] =
        average_decay * adaptive_vv_average_host[index] +
        (1.0 - average_decay) * adaptive_vv_host[index];
      adaptive_vr_average_host[index] =
        average_decay * adaptive_vr_average_host[index] +
        (1.0 - average_decay) * adaptive_vr_host[index];
      adaptive_vv_host[index] = adaptive_vv_average_host[index] / average_bias_correction;
      adaptive_vr_host[index] = adaptive_vr_average_host[index] / average_bias_correction;
    }

    for (int k = 1; k <= N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(N_f, k);
      const double gamma_k = gamma_spectrum_host[gamma_index];
      const double delta_fdt_k =
        gamma_k * adaptive_vv_host[type_offset + k] -
        adaptive_vr_host[type_offset + k];
      delta_fdt[k] = delta_fdt_k;
      delta_norm_sq += delta_fdt_k * delta_fdt_k;
    }

    for (int k = 1; k <= N_f; ++k) {
      const int gamma_index = type_offset + get_gamma_index_from_fft_bin(N_f, k);
      const double adaptive_rate_type = adaptive_rate_type_host[type];
      const double gamma_old = gamma_spectrum_host[gamma_index];
      double gamma_new = gamma_old;
      const double vv = adaptive_vv_host[type_offset + k];
      if (vv > 1.0e-30) {
        const double gamma_target = adaptive_vr_host[type_offset + k] / vv;
        if (std::isfinite(gamma_target)) {
          const double blend = fmax(0.0, fmin(1.0, adaptive_rate_type));
          double gamma_candidate = gamma_target;
          if (adaptive_tau_average > 0.0 && elapsed_average_time < adaptive_tau_average) {
            const double warmup = elapsed_average_time / adaptive_tau_average;
            gamma_candidate =
              warmup * gamma_target + (1.0 - warmup) * gamma_initial_spectrum_host[gamma_index];
          }
          gamma_new = (1.0 - blend) * gamma_old + blend * gamma_candidate;
          if (adaptive_tau_adapt > 0.0 && elapsed_average_time >= adaptive_tau_average) {
            const double gamma_decay = exp(-segment_time / adaptive_tau_adapt);
            adaptive_gamma_running_host[gamma_index] =
              gamma_decay * adaptive_gamma_running_host[gamma_index] +
              (1.0 - gamma_decay) * gamma_new;
            gamma_new = adaptive_gamma_running_host[gamma_index];
          }
        }
      }
      const double gamma_clamped =
        fmax(adaptive_gamma_min, fmin(adaptive_gamma_max, gamma_new));
      if (gamma_clamped <= adaptive_gamma_min && gamma_new < adaptive_gamma_min) {
        clamped_gamma_bins++;
      }
      if (fabs(gamma_clamped - gamma_old) > 1.0e-14) {
        gamma_changed = true;
      }
      gamma_spectrum_host[gamma_index] = gamma_clamped;
      adaptive_gamma_running_host[gamma_index] = gamma_clamped;

      if (k > 0 && k < N_f) {
        gamma_spectrum_host[type_offset + N_f - k] = gamma_clamped;
        adaptive_gamma_running_host[type_offset + N_f - k] = gamma_clamped;
      }
    }

    const int zero_index = type_offset + get_gamma_index_from_fft_bin(N_f, 0);
    const int first_index = type_offset + get_gamma_index_from_fft_bin(N_f, 1);
    gamma_spectrum_host[zero_index] = gamma_spectrum_host[first_index];
    adaptive_gamma_running_host[zero_index] = adaptive_gamma_running_host[first_index];
  }

  adaptive_update_count++;
  if (clamped_gamma_bins > 0 && !adaptive_gamma_floor_warning_issued) {
    const double frequency_conversion = 1000.0 / TIME_UNIT_CONVERSION;
    std::cout << "Warning: adQTB clamped " << clamped_gamma_bins
              << " gamma bins to the minimum floor ("
              << adaptive_gamma_min * frequency_conversion
              << " ps^-1). Consider increasing the global friction if this persists."
              << std::endl;
    adaptive_gamma_floor_warning_issued = true;
  }
  write_adaptive_qtb_diagnostics();
  if (gamma_changed) {
    write_adaptive_gamma_restart();
  }
  filter_is_dirty = gamma_changed;
}

void Ensemble_QTB::refresh_colored_random_force(const bool shift_history)
{
  gpu_refresh_qtb_random_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(),
    number_of_atoms,
    nfreq2,
    shift_history,
    time_filter_count,
    atom->type.data(),
    time_H_device.data(),
    atom->mass.data(),
    random_array_0.data(),
    random_array_1.data(),
    random_array_2.data(),
    fran.data(),
    fran.data() + number_of_atoms,
    fran.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL
}

void Ensemble_QTB::apply_qtb_force_half_step(const std::vector<Group>& group)
{
  const double dt_half = 0.5 * dt;

  if (fixed_group == -1) {
    gpu_qtb_force_half_step<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      dt_half,
      atom->mass.data(),
      atom->force_per_atom.data(),
      atom->force_per_atom.data() + number_of_atoms,
      atom->force_per_atom.data() + 2 * number_of_atoms,
      atom->velocity_per_atom.data(),
      atom->velocity_per_atom.data() + number_of_atoms,
      atom->velocity_per_atom.data() + 2 * number_of_atoms);
  } else {
    gpu_qtb_force_half_step<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      fixed_group,
      move_group,
      group[fixed_grouping_method].label.data(),
      dt_half,
      atom->mass.data(),
      atom->force_per_atom.data(),
      atom->force_per_atom.data() + number_of_atoms,
      atom->force_per_atom.data() + 2 * number_of_atoms,
      atom->velocity_per_atom.data(),
      atom->velocity_per_atom.data() + number_of_atoms,
      atom->velocity_per_atom.data() + 2 * number_of_atoms);
  }
  GPU_CHECK_KERNEL
}

void Ensemble_QTB::apply_qtb_position_half_step(const std::vector<Group>& group)
{
  const double dt_half = 0.5 * dt;

  if (fixed_group == -1) {
    gpu_qtb_position_half_step<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      dt_half,
      atom->velocity_per_atom.data(),
      atom->velocity_per_atom.data() + number_of_atoms,
      atom->velocity_per_atom.data() + 2 * number_of_atoms,
      atom->position_per_atom.data(),
      atom->position_per_atom.data() + number_of_atoms,
      atom->position_per_atom.data() + 2 * number_of_atoms);
  } else {
    gpu_qtb_position_half_step<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      fixed_group,
      move_group,
      move_velocity[0],
      move_velocity[1],
      move_velocity[2],
      group[fixed_grouping_method].label.data(),
      dt_half,
      atom->velocity_per_atom.data(),
      atom->velocity_per_atom.data() + number_of_atoms,
      atom->velocity_per_atom.data() + 2 * number_of_atoms,
      atom->position_per_atom.data(),
      atom->position_per_atom.data() + number_of_atoms,
      atom->position_per_atom.data() + 2 * number_of_atoms);
  }
  GPU_CHECK_KERNEL
}

void Ensemble_QTB::apply_qtb_ou_step()
{
  const double c1 = exp(-fric_coef * dt);
  const double c2 =
    fric_coef > 0.0 ? (1.0 - c1) / fric_coef : dt;

  gpu_apply_qtb_ou_step<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    c1,
    c2,
    atom->mass.data(),
    fran.data(),
    fran.data() + number_of_atoms,
    fran.data() + 2 * number_of_atoms,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL

  gpu_find_momentum<<<4, 1024>>>(
    number_of_atoms,
    atom->mass.data(),
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL

  gpu_correct_momentum<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

// BAOAB integration scheme:
// compute1: B -> A -> O -> A
// [force evaluation]
// compute2: B -> thermo

void Ensemble_QTB::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (use_adaptive_qtb && !adaptive_qtb_initialized) {
    initialize_adaptive_qtb();
  }

  // B: force half-step on velocities
  apply_qtb_force_half_step(group);

  // A: position half-step
  apply_qtb_position_half_step(group);

  // O: exact Ornstein-Uhlenbeck thermostat step (full dt)
  if (counter_mu == 0) {
    update_time_filter(temperature);
    refresh_colored_random_force(true);
  }
  if (use_adaptive_qtb) {
    sample_adaptive_qtb_ou_center();
  }
  apply_qtb_ou_step();

  // A: position half-step
  apply_qtb_position_half_step(group);
}

void Ensemble_QTB::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  // B: force half-step on velocities
  apply_qtb_force_half_step(group);

  find_thermo(
    true,
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);

  counter_mu = (counter_mu + 1) % alpha;
}
