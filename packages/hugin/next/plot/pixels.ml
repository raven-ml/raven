(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Box2 = Hugin_next_gg.Box2

let rgba : type a b. (a, b) Nx.t -> Nx.uint8_t =
 fun px ->
  let px =
    match Nx.shape px with [| h; w |] -> Nx.reshape [| h; w; 1 |] px | _ -> px
  in
  match Nx.dtype px with
  | Nx.UInt8 -> px
  | _ ->
      let f = Nx.cast Nx.float32 px in
      let c = (Nx.shape f).(2) in
      let hole = Nx.any ~axes:[ 2 ] ~keepdims:true (Nx.isnan f) in
      let v = Nx.round (Nx.mul_s (Nx.clamp ~min:0. ~max:1. f) 255.) in
      let v =
        Nx.where (Nx.broadcast_to (Nx.shape v) hole) (Nx.zeros_like v) v
      in
      let channel k = Nx.slice [ A; A; R (k, k + 1) ] v in
      let rgb, alpha =
        match c with
        | 1 ->
            ([ v; v; v ], Nx.where hole (Nx.zeros_like v) (Nx.full_like v 255.))
        | 3 ->
            ( [ v ],
              Nx.where hole
                (Nx.zeros_like (channel 0))
                (Nx.full_like (channel 0) 255.) )
        | _ -> ([ Nx.slice [ A; A; R (0, 3) ] v ], channel 3)
      in
      Nx.cast Nx.uint8 (Nx.concatenate ~axis:2 (rgb @ [ alpha ]))
