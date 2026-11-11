(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sentinel-1 radar backscatter: radiometrically terrain-corrected γ⁰, in
    linear power.

    Sentinel-1 carries a C-band synthetic aperture radar, which sends pulses
    of one linear {e polarisation}, horizontal or vertical, and receives the
    echo in one or both; a product holds one image for each pair it
    recorded, as [VV] and [VH]. Each value is a {e backscatter coefficient},
    the echo's power normalised by an area of the ground. {e γ⁰} (gamma
    naught) normalises it by the area perpendicular to the radar's line of
    sight; a {e radiometrically terrain-corrected} (RTC) γ⁰ uses the area that
    a digital elevation model gives each cell, so that slopes facing the
    radar or away from it no longer brighten or darken it (CEOS Analysis
    Ready Data, SAR Normalised Radar Backscatter,
    {{:https://ceos-org.github.io/ceos-ard/latest/NRB.html}}).

    {b Linear power.} A product states γ⁰ as linear power or linear amplitude,
    its square root. These providers state linear power, and their STAC items
    do not say so:
    - Microsoft's Planetary Computer, [sentinel-1-rtc]: Sentinel-1's Ground
      Range Detected products "converted to calibrated intensity" and
      terrain-corrected ("Terrain-corrected gamma naught values" in each
      polarisation's asset), float32, nodata -32768
      ({{:https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc}});
    - NASA JPL's OPERA RTC-S1, as the Alaska Satellite Facility distributes
      it: "The pixel values of the products represent gamma-0 power"
      ({{:https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/}}).

    In decibels γ⁰ is [10 · log10 γ⁰]; CEOS leaves that step to the user. An
    RTC σ⁰ (sigma naught, normalised by the area on the ground) is γ⁰ times
    the ratio of the two areas, which a product may ship as an image of its
    own (CEOS's gamma-to-sigma ratio; OPERA's RTC-STATIC "RTC ANF gamma0 to
    sigma0"): no function of the incidence angle gives it. *)

(** {1:polarisations Polarisations} *)

type polarisation =
  | HH  (** Horizontal transmit, horizontal receive. *)
  | HV  (** Horizontal transmit, vertical receive. *)
  | VH  (** Vertical transmit, horizontal receive. *)
  | VV  (** Vertical transmit, vertical receive. *)
(** The type for polarisations, transmit then receive. *)

val name : polarisation -> string
(** [name p] is [p]'s name in STAC's [sar:polarizations], as ["VH"].
    Microsoft's Planetary Computer names each asset by it in lower case. *)

val of_name : string -> polarisation option
(** [of_name s] is the polarisation whose {!name} is [s], if any. *)
