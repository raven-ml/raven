(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Pixel = struct
  type format = Rgb | Rgba | Gray

  let channels = function Rgb -> 3 | Rgba -> 4 | Gray -> 1
end

type image = {
  width : int;
  height : int;
  pixel_format : Pixel.format;
  data : (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
}

let err_data_length ~expected ~got =
  Printf.sprintf
    "Render.image: data length %d does not match width * height * channels = %d"
    got expected

let image ~width ~height ?(pixel_format = Pixel.Rgb) data =
  let expected = width * height * Pixel.channels pixel_format in
  let got = Bigarray.Array1.dim data in
  if got <> expected then invalid_arg (err_data_length ~expected ~got);
  { width; height; pixel_format; data }

let rollout env ~policy ~steps ~sink () =
  let obs, _info = Env.reset env () in
  let current_obs = ref obs in
  for _ = 1 to steps do
    let action = policy !current_obs in
    let step = Env.step env action in
    (match Env.render env with Some frame -> sink frame | None -> ());
    current_obs := step.Env.observation;
    if step.Env.terminated || step.Env.truncated then begin
      let obs, _info = Env.reset env () in
      current_obs := obs
    end
  done

let derive_id env suffix =
  match Env.id env with None -> None | Some id -> Some (id ^ suffix)

let on_render ~sink env =
  let maybe_record inner =
    match Env.render inner with Some frame -> sink frame | None -> ()
  in
  Env.wrap
    ?id:(derive_id env "/OnRender")
    ~observation_space:(Env.observation_space env)
    ~action_space:(Env.action_space env)
    ~reset:(fun inner ?options () ->
      let result = Env.reset inner ?options () in
      maybe_record inner;
      result)
    ~step:(fun inner action ->
      let s = Env.step inner action in
      maybe_record inner;
      s)
    ~render:(fun inner -> Env.render inner)
    env
