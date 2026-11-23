(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Calls over two software devices, whose outcomes the cram test reads: for
   each, whether its program compiled or ran evaluated (Rune.evaluations), and
   whether it gave eager's values. With RUNE_JIT_DEBUG set, rune prints why a
   declined program evaluates and each move between devices. *)

let report what ~eager compiled =
  let before = Rune.evaluations () in
  let got = Nx.to_array (compiled ()) in
  let how = if Rune.evaluations () = before then "compiled" else "evaluated" in
  let values =
    if got = Nx.to_array (eager ()) then "eager's values" else "other values"
  in
  Printf.printf "%s: %s, %s\n%!" what how values

let () =
  let ds =
    List.init 2 (fun k ->
        Result.get_ok (Rig.software (Printf.sprintf "declines-%d" k)))
  in
  let module S = (val Nx.devices ds) in
  let x () =
    Nx.place (S.split ~axis:0)
      (Nx.create Nx.float32 [| 4 |] [| 1.; -2.; 3.; 0.5 |])
  in
  let square a = Nx.mul a a in
  let f = Rune.jit' square in
  report "a cut argument" ~eager:(fun () -> square (x ())) (fun () -> f (x ()));
  report "a cut argument donated"
    ~eager:(fun () -> square (x ()))
    (fun () -> f (Nx.donate (x ())));
  let whole a = Nx.mul_s (Nx.place S.on a) 2. in
  report "a cut argument placed whole on each device"
    ~eager:(fun () -> whole (x ()))
    (fun () -> Rune.jit' whole (x ()));
  let index () = Nx.place S.on (Nx.scalar Nx.int32 0l) in
  let checked i x =
    Rune.switch Nx.Ptree.tensor i
      [
        (fun () ->
          Nx.check Nx.Ptree.tensor (Nx.less_s x 10.) x (fun _ _ -> Exit);
          x);
        (fun () -> x);
      ]
  in
  report "a check inside a switch's branch"
    ~eager:(fun () -> checked (index ()) (x ()))
    (fun () ->
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        checked (index ()) (x ()));
  let on k = Nx.Placement.device S.v (List.nth ds k) in
  let s () = Nx.place (on 1) (Nx.ones Nx.float32 [||]) in
  let x () = Nx.place (on 0) (Nx.ones Nx.float32 [| 4 |]) in
  let halve x = Nx.mul_s x 0.5 in
  let small x = Nx.less_s (Nx.max x) 0.01 in
  let stopped s x =
    snd (Nx.neg s, Rune.iterate' ~max:10 ~until:small halve x)
  in
  let h =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun s x -> (Nx.neg s, Rune.iterate' ~max:10 ~until:small halve x))
  in
  report "an iterate whose stop lies on another device"
    ~eager:(fun () -> stopped (s ()) (x ()))
    (fun () -> snd (h (s ()) (x ())))
