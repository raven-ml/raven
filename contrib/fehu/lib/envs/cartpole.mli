(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cart-pole balancing, the implementation of {!Fehu_envs.Cartpole}. *)

type obs = Nx.host Nx.float32_t
type act = Nx.host Nx.int32_t
type render = string

val make :
  ?render_mode:Fehu.Env.render_mode -> unit -> (obs, act, render) Fehu.Env.t
