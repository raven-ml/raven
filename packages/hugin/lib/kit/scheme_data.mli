(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The published colour tables behind {!Scheme}'s named schemes.

    A table is a string of hexadecimal digits that holds its colours as the
    pairs of their encoded sRGB components, six digits per colour and no
    separator. A table published with float components has each rounded to
    the nearest of the 256 levels a pair writes. The licence of each table
    is stated in the [.ml]. *)

(* Continuous schemes, 256 colours each *)

val viridis : string
val magma : string
val inferno : string
val plasma : string
val cividis : string
val turbo : string

(* Sequential schemes

   Element [i] is the table designed for [3 + i] classes, up to nine. *)

val blues : string array
val greens : string array
val greys : string array
val oranges : string array
val purples : string array
val reds : string array
val bugn : string array
val bupu : string array
val gnbu : string array
val orrd : string array
val pubu : string array
val pubugn : string array
val purd : string array
val rdpu : string array
val ylgn : string array
val ylgnbu : string array
val ylorbr : string array
val ylorrd : string array

(* Diverging schemes

   Element [i] is the table designed for [3 + i] classes, up to eleven. *)

val brbg : string array
val piyg : string array
val prgn : string array
val puor : string array
val rdbu : string array
val rdgy : string array
val rdylbu : string array
val rdylgn : string array
val spectral : string array

(* Qualitative schemes *)

val accent : string
val dark2 : string
val paired : string
val pastel1 : string
val pastel2 : string
val set1 : string
val set2 : string
val set3 : string
val okabe_ito : string
val tableau10 : string
