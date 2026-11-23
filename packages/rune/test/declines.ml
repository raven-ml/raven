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
  let trips = Nx.place S.on (Nx.zeros Nx.float32 [| 3 |]) in
  let scanned x =
    fst
      (Rune.scan' ~init:x
         (fun c _ ->
           Nx.check Nx.Ptree.tensor (Nx.less_s c 100.) c (fun _ _ -> Exit);
           (Nx.mul_s c 2., c))
         trips)
  in
  report "a check inside a scan's step"
    ~eager:(fun () -> scanned (x ()))
    (fun () -> Rune.jit' scanned (x ()));
  let remat_checked i x =
    Rune.switch Nx.Ptree.tensor i
      [
        (fun () ->
          Rune.remat
            Nx.Ptree.(tensor @-> returns tensor)
            (fun x ->
              Nx.check Nx.Ptree.tensor (Nx.less_s x 10.) x (fun _ _ -> Exit);
              x)
            x);
        (fun () -> x);
      ]
  in
  report "a check inside a remat inside a switch's branch"
    ~eager:(fun () -> remat_checked (index ()) (x ()))
    (fun () ->
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        remat_checked (index ()) (x ()));
  (* Additions to a total inside a construct the compiler declines with them,
     collected on the argument's placement. *)
  let t = Rune.Total.make () in
  let total f =
    let init = Nx.place (S.split ~axis:0) (Nx.zeros Nx.float32 [| 4 |]) in
    snd (Rune.Total.collect t ~init f)
  in
  let added i x =
    Rune.switch Nx.Ptree.tensor i
      [
        (fun () ->
          Rune.Total.add t x;
          x);
        (fun () -> x);
      ]
  in
  report "a total inside a switch's branch"
    ~eager:(fun () -> total (fun () -> added (index ()) (x ())))
    (fun () ->
      total (fun () ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> returns tensor)
            added (index ()) (x ())));
  let rematted x =
    Rune.remat
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x ->
        Rune.Total.add t x;
        Nx.mul_s x 2.)
      x
  in
  report "a total inside a remat's function"
    ~eager:(fun () -> total (fun () -> rematted (x ())))
    (fun () -> total (fun () -> Rune.jit' rematted (x ())));
  let checked_stop x =
    Nx.check Nx.Ptree.tensor (Nx.less_s x 100.) x (fun _ _ -> Exit);
    Nx.greater_s (Nx.max x) 10.
  in
  let doubled x =
    Rune.iterate' ~max:5 ~until:checked_stop (fun x -> Nx.mul_s x 2.) x
  in
  report "a check inside an iterate's stop"
    ~eager:(fun () -> doubled (x ()))
    (fun () -> Rune.jit' doubled (x ()));
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
