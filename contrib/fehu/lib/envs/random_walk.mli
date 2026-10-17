(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** One-dimensional random walk, the implementation of
    {!Fehu_envs.Random_walk}. *)

type obs = (float, Nx.float32_elt) Nx.t
type act = (int32, Nx.int32_elt) Nx.t
type render = string

val make :
  ?render_mode:Fehu.Env.render_mode -> unit -> (obs, act, render) Fehu.Env.t
