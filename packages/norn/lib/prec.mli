(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The limits of float dtypes. *)

val tiny : (float, 'b) Nx.dtype -> float
(** [tiny dt] is [dt]'s smallest positive normal number. Subnormals are left
    out: some devices flush them to zero. *)

val huge : (float, 'b) Nx.dtype -> float
(** [huge dt] is [dt]'s largest finite number. *)

val eps : (float, 'b) Nx.dtype -> float
(** [eps dt] is the distance from [1] to the next number of [dt]. *)
