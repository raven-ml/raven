(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Mountain car with sparse reward, the implementation of
    {!Fehu_envs.Mountain_car}. *)

type obs = Nx.host Nx.float32_t
type act = Nx.host Nx.int32_t
type render = string

val make :
  ?render_mode:Fehu.Env.render_mode -> unit -> (obs, act, render) Fehu.Env.t
