(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Standard astronomical filter bandpasses.

    Tabulated transmission curves from the
    {{:https://svo2.cab.inta-csic.es/theory/fps/} SVO Filter Profile Service}.
    Each value is a pre-built {!Photometry.bandpass}.

    {[
      let mag = Photometry.ab_mag Filters.sdss_r sed
    ]} *)

(** {1:sdss SDSS ugriz} *)

val sdss_u : Photometry.bandpass
(** [sdss_u] is the SDSS u-band (298--413 nm, 47 points). *)

val sdss_g : Photometry.bandpass
(** [sdss_g] is the SDSS g-band (363--583 nm, 89 points). *)

val sdss_r : Photometry.bandpass
(** [sdss_r] is the SDSS r-band (538--723 nm, 75 points). *)

val sdss_i : Photometry.bandpass
(** [sdss_i] is the SDSS i-band (643--863 nm, 89 points). *)

val sdss_z : Photometry.bandpass
(** [sdss_z] is the SDSS z-band (773--1123 nm, 141 points). *)

(** {1:johnson Johnson-Cousins UBVRI} *)

val johnson_u : Photometry.bandpass
(** [johnson_u] is the Johnson U-band (300--420 nm, 13 points). *)

val johnson_b : Photometry.bandpass
(** [johnson_b] is the Johnson B-band (370--560 nm, 11 points). *)

val johnson_v : Photometry.bandpass
(** [johnson_v] is the Johnson V-band (460--740 nm, 15 points). *)

val cousins_r : Photometry.bandpass
(** [cousins_r] is the Cousins R-band (540--800 nm, 53 points). *)

val cousins_i : Photometry.bandpass
(** [cousins_i] is the Cousins I-band (700--910 nm, 43 points). *)

(** {1:twomass 2MASS JHKs} *)

val twomass_j : Photometry.bandpass
(** [twomass_j] is the 2MASS J-band (1062--1450 nm, 107 points). *)

val twomass_h : Photometry.bandpass
(** [twomass_h] is the 2MASS H-band (1289--1914 nm, 58 points). *)

val twomass_ks : Photometry.bandpass
(** [twomass_ks] is the 2MASS Ks-band (1900--2399 nm, 76 points). *)

(** {1:gaia Gaia DR3} *)

val gaia_g : Photometry.bandpass
(** [gaia_g] is the Gaia DR3 G-band (330--1040 nm, 74 points). *)

val gaia_bp : Photometry.bandpass
(** [gaia_bp] is the Gaia DR3 BP-band (328--748 nm, 86 points). *)

val gaia_rp : Photometry.bandpass
(** [gaia_rp] is the Gaia DR3 RP-band (618--1076 nm, 95 points). *)

(** {1:rubin Rubin/LSST ugrizy} *)

val rubin_u : Photometry.bandpass
(** [rubin_u] is the Rubin/LSST u-band (320--409 nm, 60 points). *)

val rubin_g : Photometry.bandpass
(** [rubin_g] is the Rubin/LSST g-band (386--567 nm, 60 points). *)

val rubin_r : Photometry.bandpass
(** [rubin_r] is the Rubin/LSST r-band (537--706 nm, 60 points). *)

val rubin_i : Photometry.bandpass
(** [rubin_i] is the Rubin/LSST i-band (676--833 nm, 60 points). *)

val rubin_z : Photometry.bandpass
(** [rubin_z] is the Rubin/LSST z-band (803--935 nm, 60 points). *)

val rubin_y : Photometry.bandpass
(** [rubin_y] is the Rubin/LSST y-band (908--1099 nm, 60 points). *)

(** {1:euclid Euclid} *)

val euclid_vis : Photometry.bandpass
(** [euclid_vis] is the Euclid VIS-band (437--987 nm, 60 points). *)

val euclid_y : Photometry.bandpass
(** [euclid_y] is the Euclid NISP Y-band (933--1245 nm, 60 points). *)

val euclid_j : Photometry.bandpass
(** [euclid_j] is the Euclid NISP J-band (1141--1610 nm, 60 points). *)

val euclid_h : Photometry.bandpass
(** [euclid_h] is the Euclid NISP H-band (1480--2067 nm, 60 points). *)
