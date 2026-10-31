(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Extended grapheme clusters.

    Clusters are those of UAX #29, Unicode Text Segmentation, at Unicode
    17.0.0: its rules over the Grapheme_Cluster_Break, Indic_Conjunct_Break and
    Extended_Pictographic properties of Unicode 17.0.0, whichever version of
    Unicode the rest of brot reads. A byte that does not start a valid UTF-8
    character is a cluster of its own, of one byte. *)

val unicode_version : string
(** [unicode_version] is the Unicode version of the clusters, ["17.0.0"]. *)

val stop : string -> int -> int
(** [stop s i] is the byte index where the cluster of [s] starting at byte [i]
    ends. [i] must be the start of a cluster, such as [0] or a previous
    result, and less than [String.length s]. *)
