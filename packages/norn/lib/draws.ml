(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'u t = 'u

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let v u x =
  let lead = ref None in
  let check p t =
    let s = Nx.shape t in
    if Array.length s < 2 then
      invalid_argf "Norn.Draws.v: %s: shape [%s] has no [chain; draw] axes"
        (Nx.Ptree.Path.to_string p)
        (String.concat "; " (Array.to_list (Array.map string_of_int s)));
    match !lead with
    | None -> lead := Some (p, s.(0), s.(1))
    | Some (q, c, n) ->
        if s.(0) <> c || s.(1) <> n then
          invalid_argf
            "Norn.Draws.v: %s: %d chains of %d draws, %s: %d chains of %d draws"
            (Nx.Ptree.Path.to_string p)
            s.(0) s.(1)
            (Nx.Ptree.Path.to_string q)
            c n
  in
  Nx.Ptree.fold u (fun p t () -> check p t) x ();
  x

let ptree u = u

let map a b f d =
  let s = Nx.Ptree.(a @-> returns b) in
  Rune.vmap s (Rune.vmap s f) d

let draws u d =
  let n = Nx.Ptree.fold u (fun _ t _ -> Some (Nx.shape t).(1)) d None in
  match n with
  | Some n -> n
  | None -> invalid_arg "Norn.Draws.simulate: the draws have no tensor"

let simulate a b f k d =
  let n = draws a d in
  let chain = Rune.axis () in
  let s = Nx.Ptree.(a @-> returns b) in
  let one x =
    let i = Rune.lane_index ~axis:chain () and j = Rune.lane_index () in
    let idx = Nx.add (Nx.mul_s i (Int32.of_int n)) j in
    f (Nx.Rng.fold_in_tensor k idx) x
  in
  Rune.vmap ~axis:chain s (Rune.vmap s one) d

let append u d d' =
  let shape_s s =
    String.concat "; " (Array.to_list (Array.map string_of_int s))
  in
  Nx.Ptree.map2 u
    (fun p x y ->
      let sx = Nx.shape x and sy = Nx.shape y in
      let rest s = Array.sub s 2 (Array.length s - 2) in
      if sx.(0) <> sy.(0) || rest sx <> rest sy then
        invalid_argf "Norn.Draws.append: %s: draws of shape [%s] and [%s]"
          (Nx.Ptree.Path.to_string p)
          (shape_s sx) (shape_s sy);
      Nx.concatenate ~axis:1 [ x; y ])
    d d'

let thin u ~every d =
  if every < 1 then
    invalid_argf "Norn.Draws.thin: every = %d is not positive" every;
  Nx.Ptree.map u
    (fun _ x -> Nx.slice [ Nx.A; Nx.Rs (0, (Nx.shape x).(1), every) ] x)
    d
