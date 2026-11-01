(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Earth's ephemeris: the IAU SOFA library's iauEpv00, as ERFA's
    eraEpv00, evaluated in nx. *)

val earth :
  (Time.tdb, 'd) Time.t ->
  (float, Nx.float64_elt, 'd) Nx.t
  * (float, Nx.float64_elt, 'd) Nx.t
  * (float, Nx.float64_elt, 'd) Nx.t
  * (float, Nx.float64_elt, 'd) Nx.t
(** [earth t] is [(ph, vh, pb, vb)], [[...; 3]] each on BCRS's axes: the
    Earth's heliocentric position in au and velocity in au/d, then its
    barycentric ones, at TDB [t]. *)
