(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dust extinction laws.

    Extinction laws describe how interstellar dust attenuates and reddens light
    as a function of wavelength. A {!law} maps wavelength to the normalised
    extinction curve A{_ lambda} / A{_ V}.

    {!apply} and {!unredden} are differentiable through Rune with respect to
    [av]. The extinction curve evaluation itself (law constructors and {!curve})
    is not differentiable (scalar-level polynomial and spline evaluation). *)

(** {1:types Types} *)

type law
(** The type for extinction laws. *)

(** {1:laws Standard laws} *)

val ccm89 : rv:Nx.float64_t -> law
(** [ccm89 ~rv] is the
    {{:https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C}Cardelli, Clayton &
     Mathis (1989)} Milky Way extinction law. [rv] is the total-to-selective
    extinction ratio R{_ V} (typically 3.1).

    Valid for 0.125--3.5 {e mu}m (0.3--8.0 {e mu}m{^ -1}). Values outside this
    range are extrapolations. *)

val fitzpatrick99 : rv:Nx.float64_t -> law
(** [fitzpatrick99 ~rv] is the
    {{:https://ui.adsabs.harvard.edu/abs/1999PASP..111...63F}Fitzpatrick (1999)}
    R{_ V}-dependent Milky Way extinction law. Uses a cubic spline for
    optical/NIR and the Fitzpatrick & Massa UV parameterization.

    Valid for 0.1--3.5 {e mu}m (0.3--10.0 {e mu}m{^ -1}). *)

val odonnell94 : rv:Nx.float64_t -> law
(** [odonnell94 ~rv] is the
    {{:https://ui.adsabs.harvard.edu/abs/1994ApJ...422..158O}O'Donnell (1994)}
    Milky Way extinction law. Identical to {!ccm89} except for revised optical
    coefficients (1.1--3.3 {e mu}m{^ -1}).

    Valid for 0.125--3.5 {e mu}m. *)

val calzetti00 : law
(** [calzetti00] is the
    {{:https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C}Calzetti et al.
     (2000)} starburst attenuation law with fixed R{_ V} = 4.05.

    Valid for 0.12--2.2 {e mu}m. Values outside this range are extrapolations.
*)

(** {1:evaluation Evaluation} *)

val curve : law -> wavelength:Unit.length Unit.t -> Nx.float64_t
(** [curve law ~wavelength] is A{_ lambda} / A{_ V} at the given wavelengths.
    Not differentiable. *)

(** {1:application Application} *)

val apply : law -> av:Nx.float64_t -> 'a Spectrum.t -> 'a Spectrum.t
(** [apply law ~av spectrum] reddens [spectrum] by applying [av] magnitudes of
    V-band extinction. The spectral kind is preserved. Differentiable through
    Rune with respect to [av] and the spectrum values. *)

val unredden : law -> av:Nx.float64_t -> 'a Spectrum.t -> 'a Spectrum.t
(** [unredden law ~av spectrum] de-reddens [spectrum] by removing [av]
    magnitudes of V-band extinction. The spectral kind is preserved.
    Differentiable through Rune with respect to [av] and the spectrum values. *)
