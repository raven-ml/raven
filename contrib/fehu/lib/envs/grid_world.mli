(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Grid navigation with an obstacle, the implementation of
    {!Fehu_envs.Grid_world}.

    Rendering draws an [Image] when the render mode is [`Rgb_array] and a
    [Text] otherwise. *)

type obs = (int32, Nx.int32_elt) Nx.t
type act = (int32, Nx.int32_elt) Nx.t
type render = Text of string | Image of Fehu.Render.image

val make :
  ?render_mode:Fehu.Env.render_mode -> unit -> (obs, act, render) Fehu.Env.t
