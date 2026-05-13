.. _kw_ensemble_qtb:
.. index::
   single: nvt_qtb (keyword in run.in)
   single: npt_qtb (keyword in run.in)
   single: Quantum Thermal Bath

:attr:`ensemble` (QTB)
======================

The variants of the :attr:`ensemble` keyword described on this page implement the Quantum Thermal Bath (QTB) method [Dammak2009]_.
The QTB thermostat is a Langevin-type thermostat that uses colored noise with a quantum Bose-Einstein energy spectrum instead of classical white noise.
This allows approximate inclusion of nuclear quantum effects (zero-point energy and quantum heat capacity) in otherwise classical MD simulations.

The :attr:`npt_qtb` variant combines the QTB thermostat with the Parrinello-Rahman (MTTK) barostat [Martyna1994]_.


Syntax
------

:attr:`nvt_qtb`
^^^^^^^^^^^^^^^

Run an NVT simulation with the QTB thermostat::

    ensemble nvt_qtb <T_1> <T_2> <T_coup> [f_max <value>] [N_f <value>] [adaptive <off|on>] [adapt_window <value>] [adapt_rate <value>] [theta_correction <0|1>]

* :attr:`<T_1>` and :attr:`<T_2>`: Initial and final target temperature (K). The target temperature varies linearly during the run.
* :attr:`<T_coup>`: Thermostat coupling parameter (in units of timestep). Controls the friction coefficient: :math:`\gamma = 1 / (\text{T\_coup} \times dt)`.
* :attr:`f_max`: (Optional, default 200) Maximum frequency of the QTB filter in ps\ :sup:`-1`. Should be larger than the highest phonon frequency in the system.
* :attr:`N_f`: (Optional, default 100) Number of frequency points in the filter. The filter uses :math:`2 N_f` points total.
* :attr:`adaptive`: (Optional, default ``off``) Enable adaptive QTB. Accepted values are ``off``, ``on``, ``0``, and ``1``.
* :attr:`adapt_window`: (Optional) Adaptive segment length in units of timestep. For example, ``adapt_window 4000`` updates the system-bath coupling spectrum after every 4000 MD steps.
* :attr:`adapt_rate`: (Optional, default 0.1) Adaptive update coefficient. Most users should keep the default.
* :attr:`theta_correction`: (Optional, default 1) Enable the finite-friction corrected QTB energy density. Setting this to 0 uses the raw Bose-Einstein spectrum.

:attr:`npt_qtb`
^^^^^^^^^^^^^^^

Run an NPT simulation with the QTB thermostat and Parrinello-Rahman (MTTK) barostat::

    ensemble npt_qtb <direction> <p_1> <p_2> temp <T_1> <T_2> tperiod <tau_T> pperiod <tau_p> [f_max <value>] [N_f <value>] [adaptive <off|on>] [adapt_window <value>] [adapt_rate <value>] [theta_correction <0|1>]

Pressure control parameters:

* :attr:`<direction>`: One or more of ``iso``, ``aniso``, ``tri``, ``x``, ``y``, ``z``. Same syntax as :ref:`npt_mttk <mttk>`.
* :attr:`<p_1>` and :attr:`<p_2>`: Initial and final target pressure (GPa).

Temperature and coupling parameters:

* :attr:`temp <T_1> <T_2>`: Initial and final target temperature (K).
* :attr:`tperiod <tau_T>`: QTB thermostat coupling period (in units of timestep). Controls friction: :math:`\gamma = 1 / (\text{tperiod} \times dt)`.
* :attr:`pperiod <tau_p>`: Barostat coupling period (in units of timestep, must be :math:`\geq 200`).

QTB-specific optional parameters (same as :attr:`nvt_qtb`):

* :attr:`f_max`: Maximum frequency (ps\ :sup:`-1`, default 200).
* :attr:`N_f`: Number of frequency points (default 100).
* :attr:`adaptive`: Enable adaptive QTB (default ``off``).
* :attr:`adapt_window`: Adaptive segment length in units of timestep.
* :attr:`adapt_rate`: Adaptive update coefficient (default 0.1).
* :attr:`theta_correction`: Enable the finite-friction corrected QTB energy density (default 1).

Advanced adQTB parameters
-------------------------

The following options are mainly intended for convergence tests and method development:

* :attr:`adapt_tau_avg`: Averaging time for the adaptive ratio estimator, in units of timestep. If omitted, GPUMD uses 10 adaptive segments.
* :attr:`adapt_tau_adapt`: Additional smoothing time for the adapted system-bath coupling spectrum, in units of timestep.
* :attr:`adapt_smooth`: Frequency-window smoothing width for adaptive spectra, in ps\ :sup:`-1`.
* :attr:`adapt_rate_type`: Per-atom-type override of :attr:`adapt_rate`. Use ``adapt_rate_type <type_id> <value>``.


Examples
--------

NVT-QTB
^^^^^^^^

.. code-block:: rst

    ensemble nvt_qtb 300 300 100

Run at 300 K with QTB thermostat. The coupling parameter is 100 timesteps.

.. code-block:: rst

    ensemble nvt_qtb 300 300 100 f_max 150 N_f 200

Same as above but with custom filter parameters.

.. code-block:: rst

    ensemble nvt_qtb 300 300 100 f_max 200 N_f 100 adaptive on adapt_window 4000

Run adaptive QTB and update the system-bath coupling spectrum every 4000 timesteps.

.. code-block:: rst

    ensemble nvt_qtb 10 10 100 f_max 20.371833 N_f 256 adaptive on adapt_window 4000

For a 12,000,000-step run, this gives 3000 adaptive segments of 4000 steps each. Statistical analysis can then discard the first half of the trajectory and use the last 1500 segments.

NPT-QTB
^^^^^^^^

.. code-block:: rst

    ensemble npt_qtb iso 0 0 temp 300 300 tperiod 100 pperiod 1000

Run at 300 K and 0 GPa with isotropic pressure control.

.. code-block:: rst

    ensemble npt_qtb aniso 0 0 temp 300 300 tperiod 100 pperiod 1000 f_max 200 N_f 100

Anisotropic pressure control with explicit QTB parameters.

.. code-block:: rst

    ensemble npt_qtb x 5 5 y 0 0 z 0 0 temp 300 300 tperiod 100 pperiod 1000

Apply 5 GPa along x and 0 GPa along y and z.

.. code-block:: rst

    ensemble npt_qtb iso 0 0 temp 300 300 tperiod 250 pperiod 1000 f_max 360 N_f 420 adaptive on adapt_window 5000

Isotropic NPT-QTB with adaptive QTB enabled.


Notes
-----

* The QTB method generates colored noise whose power spectrum matches the quantum energy distribution :math:`E(\omega) = \hbar\omega[\frac{1}{2} + n_{BE}(\omega, T)]`, where :math:`n_{BE}` is the Bose-Einstein distribution.
* The kinetic temperature reported in ``thermo.out`` will be higher than the target temperature due to zero-point energy contributions. This is expected behavior.
* For liquid water at 300 K, the kinetic temperature is typically around 1000-1100 K.
* The :attr:`npt_qtb` ensemble uses the MTTK (Martyna-Tuckerman-Tobias-Klein) integrator for pressure control, which is the same as :ref:`npt_mttk <mttk>` but with the Nosé-Hoover chain thermostat replaced by the QTB thermostat.
* The :attr:`f_max` parameter should be set larger than the highest phonon frequency in the system. For water, 200 ps\ :sup:`-1` is sufficient.
* When adaptive QTB is enabled, GPUMD writes ``qtb_adaptive_gamma.out``, ``qtb_adaptive_fdt.out``, and ``qtb_theta_correction.out``. It also writes ``qtb_gamma_restart.out`` and loads it automatically on the next QTB initialization in the same directory.


References
----------

.. [Dammak2009] H. Dammak, Y. Chalopin, M. Laroche, M. Hayoun, and J.-J. Greffet, *Quantum Thermal Bath for Molecular Dynamics Simulation*, Phys. Rev. Lett. **103**, 190601 (2009).
