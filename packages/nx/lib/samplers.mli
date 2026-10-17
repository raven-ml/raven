(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Random sampling: the implementation of {!Nx.Rng} and of {!Nx}'s keyless
    samplers.

    One Threefry-2x32 generator has two front ends. The samplers of {!Rng}
    take a key and are pure functions of it. The keyless samplers draw the
    next key of the scope around them ({!Rng.next_key}) and call the same
    samplers. A function here that takes a context takes it first and makes
    its key at it, and {!Nx} fixes it to {!Placement.host}. *)

open Frontend

module Rng : sig
  type key = int32_t
  (** {!Nx.Rng.t}, which makes it private. *)

  val key : context -> int -> key
  (** [key ctx seed] is {!Nx.Rng.key}[ seed] at [ctx]. *)

  val of_tensor : int32_t -> key
  val split : ?n:int -> key -> key array
  val split_batch : n:int -> key -> key
  val fold_in : key -> int -> key
  val fold_in_tensor : key -> int32_t -> key

  (** {1:samplers Samplers} *)

  val bits : key -> int array -> int32_t
  val uniform : key -> (float, 'b) dtype -> int array -> (float, 'b) t
  val normal : key -> (float, 'b) dtype -> int array -> (float, 'b) t

  val randint : key -> ?low:int -> high:int -> int array -> int32_t
  val bernoulli : key -> (float, 'b) t -> (bool, bool_elt) t

  val truncated_normal :
    key -> (float, 'b) t -> (float, 'b) t -> (float, 'b) t

  val gumbel : key -> (float, 'b) dtype -> int array -> (float, 'b) t
  val exponential : key -> (float, 'b) dtype -> int array -> (float, 'b) t
  val gamma : key -> (float, 'b) t -> (float, 'b) t
  val beta : key -> (float, 'b) t -> (float, 'b) t -> (float, 'b) t
  val dirichlet : key -> (float, 'b) t -> (float, 'b) t
  val von_mises : key -> (float, 'b) t -> (float, 'b) t
  val poisson : key -> (float, 'b) t -> int32_t
  val binomial : key -> int32_t -> (float, 'b) t -> int32_t
  val categorical : key -> ?axis:int -> (float, 'b) t -> int64_t
  val permutation : key -> int -> int64_t
  val shuffle : key -> ('a, 'b) t -> ('a, 'b) t

  (** {1:scope Scopes} *)

  val with_key : key -> (unit -> 'a) -> 'a
  val with_root : (unit -> key) -> (unit -> 'a) -> 'a

  val next_key : context -> key
  (** [next_key ctx] is {!Nx.Rng.next_key}[ ()]. Outside any scope, it is the
      next key of the calling domain's generator, which system entropy seeds
      at [ctx] on its first draw. *)

  val next_root : context -> unit -> key
  (** [next_root ctx] is {!Nx.Rng.next_root}[ ()]. Made outside any scope,
      its function answers every call with one key: the next key of the
      generator of the domain that calls it first, seeded as {!next_key}
      seeds it, taken at that call. A scope never answers it, so a root made
      outside a scope and first called inside {!with_key} draws from the
      domain's generator. *)
end

(** {1:keyless Keyless samplers}

    Each draws {!Rng.next_key}[ ctx] and calls its namesake in {!Rng}, or
    {!Rng.uniform} for [rand] and {!Rng.normal} for [randn]. *)

val rand : context -> (float, 'b) dtype -> int array -> (float, 'b) t
val randn : context -> (float, 'b) dtype -> int array -> (float, 'b) t
val randint : context -> ?low:int -> high:int -> int array -> int32_t
val bernoulli : context -> (float, 'b) t -> (bool, bool_elt) t
val permutation : context -> int -> int64_t
val shuffle : context -> ('a, 'b) t -> ('a, 'b) t
val categorical : context -> ?axis:int -> (float, 'b) t -> int64_t

val truncated_normal :
  context -> (float, 'b) t -> (float, 'b) t -> (float, 'b) t
