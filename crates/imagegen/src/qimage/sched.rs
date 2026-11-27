//! Flow-matching Euler schedule (diffusers `FlowMatchEulerDiscreteScheduler`
//! as `QwenImage21Pipeline` drives it).
//!
//! Turbo saves its 8 sigmas in `model_index.json` `sample_sigmas`; with the
//! scheduler's `use_dynamic_shifting: false` and `shift: 1` they are used as
//! given (a step count does not change them). The base model (no
//! `sample_sigmas`) takes `linspace(1, 1 / n, n)` and the exponential shift
//! with `mu` interpolated from the image's token count. A terminal 0 is
//! appended. Computed in f32 like the scheduler's tensors.

use super::config::SchedCfg;

/// The extra step of the default Turbo schedule: Turbo's last saved sigma
/// is 0.41, and the jump from there to 0 smooths fine texture away (waxy
/// skin, soft fur); one more step at 0.3 keeps the composition (1/16-scale
/// RMS change 0.009 vs 0.06-0.15 for a resampled 12-25 step curve) and adds
/// ~9 % high-frequency detail for one step's time. 0.2 / 0.1: harsher /
/// weaker (`docs/image.md` "Steps").
pub const REFINE_SIGMA: f32 = 0.3;

pub struct Schedule {
	/// `n + 1` sigmas, the last 0.
	pub sigmas: Vec<f32>,
}

impl Schedule {
	pub fn new(
		cfg: &SchedCfg,
		sample: Option<&[f64]>,
		steps: usize,
		image_tokens: usize,
	) -> Schedule {
		let mut s: Vec<f32> = match sample {
			Some(v) => v.iter().map(|&x| x as f32).collect(),
			None => (0..steps)
				.map(|i| {
					let a = 1.0f64;
					let b = 1.0 / steps as f64;
					(a + (b - a) * i as f64 / (steps.max(2) - 1) as f64) as f32
				})
				.collect(),
		};
		if cfg.dynamic {
			let m = (cfg.max_shift - cfg.base_shift) / (cfg.max_seq - cfg.base_seq);
			let mu = image_tokens as f64 * m + (cfg.base_shift - m * cfg.base_seq);
			let e = mu.exp();
			for v in &mut s {
				*v = (e / (e + (1.0 / *v as f64 - 1.0))) as f32;
			}
		} else {
			let sh = cfg.shift as f32;
			for v in &mut s {
				*v = sh * *v / (1.0 + (sh - 1.0) * *v);
			}
		}
		if let Some(term) = cfg.shift_terminal {
			let last = *s.last().unwrap() as f64;
			let k = (1.0 - last) / (1.0 - term);
			for v in &mut s {
				*v = (1.0 - (1.0 - *v as f64) / k) as f32;
			}
		}
		s.push(0.0);
		Schedule { sigmas: s }
	}

	/// A job's schedule for `steps` (0 = the default). With `sample_sigmas`
	/// (Turbo): the default is those sigmas plus one step at
	/// [`REFINE_SIGMA`] inside the last interval (9 steps), `sample.len()`
	/// gives them as saved, any other count resamples the saved curve
	/// ([`Schedule::resample`]: changes the composition, harsher from ~12
	/// on). Without: `steps` (default 40) as [`Schedule::new`].
	pub fn for_steps(
		cfg: &SchedCfg,
		sample: Option<&[f64]>,
		steps: usize,
		image_tokens: usize,
	) -> Schedule {
		let Some(v) = sample else {
			return Schedule::new(cfg, None, if steps == 0 { 40 } else { steps }, image_tokens);
		};
		let mut s = Schedule::new(cfg, sample, steps, image_tokens);
		let n = v.len();
		if steps == 0 || steps == n + 1 {
			let last = s.sigmas[n - 1];
			if REFINE_SIGMA < last {
				s.sigmas.insert(n, REFINE_SIGMA);
			}
			s
		} else if steps == n {
			s
		} else {
			s.resample(steps)
		}
	}

	/// The same curve at `n` steps: the sigmas (terminal 0 included) as a
	/// piecewise-linear function of the step fraction, sampled at `j / n`.
	pub fn resample(&self, n: usize) -> Schedule {
		let m = self.steps();
		let mut s: Vec<f32> = (0..n)
			.map(|j| {
				let u = j as f64 * m as f64 / n as f64;
				let k = (u.floor() as usize).min(m - 1);
				let f = u - k as f64;
				(self.sigmas[k] as f64 * (1.0 - f) + self.sigmas[k + 1] as f64 * f) as f32
			})
			.collect();
		s.push(0.0);
		Schedule { sigmas: s }
	}

	pub fn steps(&self) -> usize {
		self.sigmas.len() - 1
	}

	/// The transformer's timestep at step `i`: `sigma * 1000 / 1000` in f32
	/// (the pipeline hands the scheduler's `timesteps / 1000`).
	pub fn t(&self, i: usize, train_steps: f64) -> f32 {
		let ts = self.sigmas[i] * train_steps as f32;
		ts / train_steps as f32
	}

	/// One Euler step: `x += (sigma[i+1] - sigma[i]) v`.
	pub fn step(&self, i: usize, x: &mut [f32], v: &[f32]) {
		let d = self.sigmas[i + 1] - self.sigmas[i];
		for (a, &b) in x.iter_mut().zip(v) {
			*a += d * b;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const TURBO: [f64; 8] = [
		1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
	];

	fn cfg() -> SchedCfg {
		SchedCfg {
			train_steps: 1000.0,
			shift: 1.0,
			dynamic: false,
			base_seq: 256.0,
			max_seq: 8192.0,
			base_shift: 0.5,
			max_shift: 0.9,
			shift_terminal: None,
		}
	}

	#[test]
	fn turbo_default_adds_the_refinement_step() {
		let s = Schedule::for_steps(&cfg(), Some(&TURBO), 0, 4096);
		assert_eq!(s.steps(), 9);
		assert_eq!(s.sigmas[7], 0.414568);
		assert_eq!(s.sigmas[8], REFINE_SIGMA);
		assert_eq!(s.sigmas[9], 0.0);
		assert_eq!(
			Schedule::for_steps(&cfg(), Some(&TURBO), 9, 4096).sigmas,
			s.sigmas
		);
	}

	#[test]
	fn turbo_eight_is_the_saved_schedule() {
		let s = Schedule::for_steps(&cfg(), Some(&TURBO), 8, 4096);
		let want: Vec<f32> = TURBO.iter().map(|&x| x as f32).chain([0.0]).collect();
		assert_eq!(s.sigmas, want);
	}

	#[test]
	fn other_counts_resample_the_curve() {
		let s = Schedule::for_steps(&cfg(), Some(&TURBO), 16, 4096);
		assert_eq!(s.steps(), 16);
		assert_eq!(s.sigmas[0], 1.0);
		assert_eq!(s.sigmas[2], 0.978453);
		assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
	}

	#[test]
	fn base_model_default_is_forty() {
		assert_eq!(Schedule::for_steps(&cfg(), None, 0, 4096).steps(), 40);
		assert_eq!(Schedule::for_steps(&cfg(), None, 20, 4096).steps(), 20);
	}
}
