Movement metrics
================

MyoInteract simulates users of interaction techniques with a biomechanical model.
Success rate and return say whether a policy solves a task, not how it moves.
``myosuite.utils.movement_metrics`` adds the standard HCI and motor-control measures of
point-to-point movements: movement time, accuracy, velocity profile, smoothness, Fitts'
law and effort. ``myosuite.envs.myo.myouser.pointing_metrics`` applies them to MyoUser
evaluation rollouts, one trial per target of a (sequential) pointing or button task. The
metrics module uses only NumPy and SciPy.

Usage
-----

Log the metrics during and after training (off by default)::

   python myosuite/envs/myo/myouser/train_jax_ppo.py run.movement_metrics=true

Each checkpoint evaluation then logs ``eval_movement/<key>`` to Weights & Biases next to
the ``eval/`` metrics, and the final evaluation writes ``movement_metrics.json`` (summary
and per-target trials) to the run's log directory and logs ``final_policy/movement/<key>``.
Training is unchanged.

Score rollouts in a notebook, for example after loading a checkpoint as in
``myointeract_ui.ipynb``:

.. code-block:: python

   from myosuite.envs.myo.myouser.evaluate import evaluate_policy
   from myosuite.envs.myo.myouser.pointing_metrics import rollout_trials, summarize_trials

   rollouts, _ = evaluate_policy(eval_env=eval_env, jit_inference_fn=jit_inference_fn,
                                 jit_reset=jit_reset, jit_step=jit_step,
                                 n_episodes=20, include_reset_state=True)
   trials = rollout_trials(rollouts, eval_env)   # trials[episode][target]: dict per trial
   summary = summarize_trials(trials)            # means, per-target means, Fitts' law

``include_reset_state=True`` starts each rollout at the reset, so that the first
target's trial starts there; the other outputs of ``evaluate_policy`` are unchanged.

Use the functions directly on any sampled trajectory, for example an ISO 9241-411
analysis over distance x width conditions:

.. code-block:: python

   from myosuite.utils import movement_metrics as mm

   trial = mm.pointing_trial_metrics(positions, target, radius, dt, selected=True)
   eff = mm.effective_parameters(starts, targets, endpoints)       # one condition
   tp = mm.throughput(ide_per_condition, mt_per_condition)          # bits/s
   fit = mm.fitts_regression(ide_per_condition, mt_per_condition)   # MT = a + b ID
   summary = mm.fitts_summary(trials, conditions=labels)            # all of the above

Per-trial metrics
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Key
     - Definition
     - Reference
   * - ``success``
     - The target was selected (dwell completed or button pressed).
     - —
   * - ``trial_time``
     - Target activation (the previous selection, or the reset) to its selection (s);
       the movement time of Fitts' law tasks, which ends with the selection.
     - MacKenzie 1992
   * - ``distance``, ``width``, ``index_of_difficulty``
     - D from the fingertip at activation to the target centre, W = 2 x radius,
       ``ID = log2(D / W + 1)`` (bits).
     - MacKenzie 1992; MacKenzie 2018, Eq. 17.6
   * - ``effective_amplitude``, ``endpoint_deviation``
     - ae and the signed overshoot dx of the selection point along the start-to-target
       axis.
     - Soukoreff & MacKenzie 2004; ISO 9241-411
   * - ``final_error``
     - Distance from the selection point to the target centre (m).
     - —
   * - ``time_to_target``
     - First entry into the target, from activation.
     - —
   * - ``time_to_acquire``
     - Entry that starts the dwell that selects the target, from activation
       (``trial_time`` without the dwell).
     - —
   * - ``target_entries``
     - Number of entries into the target; minus one is the target re-entry count.
     - MacKenzie, Kauppinen & Silfverberg 2001
   * - ``movement_time``
     - Offset minus onset, both at 5% of peak speed (linearly interpolated), including
       corrective submovements.
     - Schot, Brenner & Smeets 2010
   * - ``peak_speed``
     - Maximum tangential fingertip speed (m/s).
     - —
   * - ``time_to_peak_ratio``
     - Acceleration time over movement time; 0.5 for a symmetric profile.
     - Nagasaki 1989
   * - ``speed_peaks``
     - Speed peaks (``scipy.signal.find_peaks``, height and prominence 5% of peak
       speed), a count of submovements.
     - Rohrer et al. 2002
   * - ``ldlj``
     - Log dimensionless jerk ``-ln(T^3 / v_peak^2 * integral ||x'''||^2 dt)`` over the
       movement segment; ``normalization="amplitude"`` gives the ``T^5 / A^2`` form.
     - Balasubramanian et al. 2012, 2015; Hogan & Sternad 2009
   * - ``sparc``
     - Spectral arc length of the speed profile (10 Hz maximum cutoff, 0.05 threshold,
       padding level 4).
     - Balasubramanian et al. 2015
   * - ``straightness``
     - Path length over start-to-end distance (path-length ratio); 1 is straight.
     - Schwarz et al. 2019
   * - ``effort``
     - Mean squared muscle activation.
     - Ackermann & van den Bogert 2010

Summary metrics
---------------

``summarize_trials`` reports ``trials`` (count), the means of the per-trial metrics over
all trials (so ``success`` is the selection rate; undefined values are skipped),
``target_<k>/trial_time``, ``target_<k>/index_of_difficulty`` and
``target_<k>/success`` per target of the sequence, and the Fitts' law summary of
``fitts_summary``:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Key
     - Definition
     - Reference
   * - ``fitts_trials``, ``fitts_conditions``
     - Selected trials used; conditions (groups of at least 2 trials) with a defined
       IDe (We > 0 and De > 0, so not for selections that end behind the start).
     - —
   * - ``fitts_intercept``, ``fitts_slope``, ``fitts_r_squared``
     - Least-squares ``MT = a + b ID`` over the trials (nominal ID; at least 3 trials).
     - Fitts 1954; MacKenzie 1992
   * - ``fitts_effective_index_of_difficulty``
     - Mean over conditions of ``IDe = log2(De / We + 1)``, with ``We = 4.133 SD(dx)``
       and ``De = mean(ae)``.
     - Soukoreff & MacKenzie 2004; ISO 9241-411
   * - ``fitts_effective_width_ratio``
     - Mean over conditions of We / W.
     - —
   * - ``fitts_throughput``
     - Mean over conditions of ``IDe / MT`` (bits/s).
     - Soukoreff & MacKenzie 2004; MacKenzie 2018, Eq. 17.10
   * - ``fitts_effective_intercept``, ``fitts_effective_slope``,
       ``fitts_effective_r_squared``
     - Least-squares ``MT = a + b IDe`` over the conditions (at least 3).
     - Fitts 1954; Soukoreff & MacKenzie 2004

Other functions of the module: ``minimum_jerk`` (Flash & Hogan 1985, also used by
``myosuite.utils.min_jerk``), ``two_thirds_power_law`` (Lacquaniti, Terzuolo & Viviani
1983), ``effective_parameters``, ``throughput``, ``fitts_regression`` and
``fitts_conditions``.

Conventions
-----------

* **Sampling.** Trajectories are sampled every control step, ``dt = ctrl_dt`` (0.05 s by
  default, 20 Hz). SPARC's 10 Hz cutoff is then the Nyquist frequency, and a trial of
  0.5 s has 11 samples, so smoothness values are coarse; compare them across policies
  with the same ``ctrl_dt``.
* **Derivatives.** By default, velocity, acceleration and jerk come from a quintic
  interpolating spline (``scipy.interpolate.make_interp_spline``). It is exact for
  minimum-jerk movements and suits noise-free simulation. With motor noise, pass
  ``savgol=(window, polyorder)`` to use Savitzky-Golay smoothing.
* **Segmentation.** ``info["phase"]`` after a step is the target pursued next. Target k's
  trial runs from the step at which it became the phase to the step at which it is
  selected (the phase changes, or the episode ends with ``done``), or to the end of the
  episode. Targets after an unselected one are never activated and have no trial.
* **Inside and dwell.** "Inside" is the env's own test (``info["steps_inside_target"]``),
  so a button counts as inside while pressed. The dwell of each target comes from the
  env (``dwell_duration``; one step for buttons), and the env counts the activation
  sample, so ``trial_time - time_to_acquire`` is one step shorter than the dwell.
* **Buttons.** W = 2 x the button's half-width and D is measured to the button's centre,
  so Fitts' law values of button targets are approximate.
* **Fitts' law conditions.** ISO 9241-411 computes We and IDe per condition (one D and
  one W). MyoUser targets are sampled at random, so no two trials share a condition.
  Without ``conditions``, ``fitts_summary`` sorts the trials by nominal ID and splits them
  into ``n_bins`` (default 3) groups of nearly equal size. Within a group, ae and dx are
  divided by each trial's nominal width before De and We are computed: Fitts' law depends
  on D / W only, and with one width per group this is exactly the ISO computation. Pass
  ``conditions`` (one label per trial) for tasks with fixed D x W conditions. The movement
  time is ``trial_time`` (``time_key="time_to_acquire"`` leaves out the dwell).

Validation
----------

``myosuite/tests/test_movement_metrics.py`` checks the functions against known values:

* Minimum jerk: peak speed ``1.875 D / T``, time-to-peak ratio 0.5, squared jerk 720 with
  the amplitude normalisation and ``720 / 1.875^2 = 204.8`` with the peak-speed
  normalisation, LDLJ ``-ln 204.8``, all from sampled trajectories.
* The SPARC and LDLJ doctests of the reference code of Balasubramanian et al.
  (github.com/siva82kb/SPARC).
* Onset detection on a noisy minimum-jerk movement, effective width of Gaussian
  endpoints, a hand-computed throughput, recovery of a known Fitts' law line,
  straightness of a line and a semicircle, two speed peaks for two overlapping
  submovements, and the exact two-thirds exponent of a harmonic ellipse.
* Pointing trials: D, W, ID, ae and dx of a minimum-jerk trial with a known overshoot;
  ``fitts_conditions`` equal to ``effective_parameters`` for one width per condition and
  invariant to the scale of a condition; recovery of a known ``MT = a + b IDe`` law and
  its throughput.

``myosuite/tests/test_myouser_pointing_metrics.py`` checks the segmentation on a
synthetic two-target episode scored with MyoUserUniversal's rules and on a real
MyoUserUniversal rollout through ``evaluate_policy``.

References
----------

* Ackermann, M. & van den Bogert, A. J. (2010). Optimality principles for model-based
  prediction of human gait. *J. Biomech.* 43(6), 1055-1060.
* Balasubramanian, S., Melendez-Calderon, A. & Burdet, E. (2012). A robust and sensitive
  metric for quantifying movement smoothness. *IEEE Trans. Biomed. Eng.* 59(8), 2126-2136.
* Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. (2015). On the
  analysis of movement smoothness. *J. NeuroEng. Rehabil.* 12, 112.
* Fitts, P. M. (1954). The information capacity of the human motor system in controlling
  the amplitude of movement. *J. Exp. Psychol.* 47(6), 381-391.
* Flash, T. & Hogan, N. (1985). The coordination of arm movements: an experimentally
  confirmed mathematical model. *J. Neurosci.* 5(7), 1688-1703.
* Hogan, N. & Sternad, D. (2009). Sensitivity of smoothness measures to movement
  duration, amplitude, and arrests. *J. Mot. Behav.* 41(6), 529-534.
* ISO 9241-411 (2012). Ergonomics of human-system interaction, Part 411: Evaluation
  methods for the design of physical input devices.
* Lacquaniti, F., Terzuolo, C. & Viviani, P. (1983). The law relating the kinematic and
  figural aspects of drawing movements. *Acta Psychol.* 54, 115-130.
* MacKenzie, I. S. (1992). Fitts' law as a research and design tool in human-computer
  interaction. *Hum.-Comput. Interact.* 7, 91-139.
* MacKenzie, I. S. (2018). Fitts' law. In K. L. Norman & J. Kirakowski (Eds.), *Handbook
  of Human-Computer Interaction*, 349-370. Wiley.
* MacKenzie, I. S., Kauppinen, T. & Silfverberg, M. (2001). Accuracy measures for
  evaluating computer pointing devices. *Proc. CHI 2001*, 9-16.
* Nagasaki, H. (1989). Asymmetric velocity and acceleration profiles of human arm
  movements. *Exp. Brain Res.* 74, 319-326.
* Rohrer, B. et al. (2002). Movement smoothness changes during stroke recovery.
  *J. Neurosci.* 22(18), 8297-8304.
* Schot, W. D., Brenner, E. & Smeets, J. B. J. (2010). Robust movement segmentation by
  combining multiple sources of information. *J. Neurosci. Methods* 187, 147-155.
* Schwarz, A., Kanzler, C. M., Lambercy, O., Luft, A. R. & Veerbeek, J. M. (2019).
  Systematic review on kinematic assessments of upper limb movements after stroke.
  *Stroke* 50(3), 718-727.
* Soukoreff, R. W. & MacKenzie, I. S. (2004). Towards a standard for pointing device
  evaluation, perspectives on 27 years of Fitts' law research in HCI. *Int. J.
  Hum.-Comput. Stud.* 61(6), 751-789.
