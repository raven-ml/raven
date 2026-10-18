(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.check: the first false flag in C order raises the caller's exception with
   its index and the data's elements there, each array broadcast to the flags'
   shape; at the call on the host, at the next wait on another set. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Ptree

let m = Nx_support.memory

module S = (val Nx.devices [ m 0 ])

exception At of int array * float * int32 * float
exception Flag of int array
exception Element of string

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let numel s = Array.fold_left ( * ) 1 s

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

(* The index of the flat position [i] of a value of shape [s], in C order. *)
let index s i =
  let idx = Array.make (Array.length s) 0 and k = ref i in
  for a = Array.length s - 1 downto 0 do
    idx.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  idx

(* Flags of a shape with some extents of 0 or 1, mostly true. *)
let flags =
  let open Gen in
  (let* s =
     array ~size:(int_range 0 3)
       (frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 4) ])
   in
   let+ ok =
     array
       ~size:(constant (numel s))
       (frequency [ (6, constant true); (1, constant false) ])
   in
   (s, ok))
  |> with_pp (fun ppf (s, ok) ->
      Format.fprintf ppf "%a %s" pp_ints s
        (String.concat ""
           (Array.to_list (Array.map (fun b -> if b then "1" else "0") ok))))

(* Data of three arrays: one of the flags' shape, a 0-d one, and one of the
   flags' last axis alone, broadcast along the others. *)
let law (s, ok) =
  let n = numel s and r = Array.length s in
  let first = Array.find_index not ok in
  cover "passes" (Option.is_none first && n > 0);
  cover "fails" (Option.is_some first);
  cover "no element" (n = 0);
  cover "rank 0" (r = 0);
  cover "fails past the first element" (Option.value ~default:0 first > 0);
  let xs = Array.init n float_of_int in
  let last = if r = 0 then [||] else [| s.(r - 1) |] in
  let data =
    ( Nx.create Nx.float32 s xs,
      ( Nx.create Nx.int32 [||] [| 7l |],
        Nx.create Nx.float32 last
          (Array.init (numel last) (fun j -> 100. +. float_of_int j)) ) )
  in
  let desc = P.(pair tensor (pair tensor tensor)) in
  let fail i (a, (b, c)) = At (i, Nx.item [] a, Nx.item [] b, Nx.item [] c) in
  let ok = Nx.create Nx.bool s ok in
  match first with
  | None -> Nx.check desc ok data fail
  | Some j ->
      let i = index s j in
      let column = if r = 0 then 100. else 100. +. float_of_int i.(r - 1) in
      raises (At (i, xs.(j), 7l, column)) (fun () -> Nx.check desc ok data fail)

let host =
  group "on the host"
    [
      prop "the first false flag raises at the call with the data there" flags
        law;
      cases
        ~name:(fun (D.Any dt) -> D.name dt)
        "the data's elements reach the exception at every dtype" D.all
        (fun (D.Any dt) ->
          let vs = [| D.zero dt; D.one dt; D.of_float dt 3. |] in
          let data = Nx.create dt [| 3 |] vs in
          let ok = Nx.create Nx.bool [| 3 |] [| true; true; false |] in
          let show v = Format.asprintf "%a" (D.pp_value dt) v in
          raises
            (Element (show vs.(2)))
            (fun () ->
              Nx.check P.tensor ok data (fun _ x ->
                  Element (show (Nx.item [] x)))));
      test "empty flags pass, and a check without data passes unit" (fun () ->
          let data = Nx.create Nx.float32 [| 2; 0 |] [||] in
          Nx.check P.tensor
            (Nx.create Nx.bool [| 2; 0 |] [||])
            data
            (fun _ _ -> Failure "an empty check failed");
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          raises (Flag [| 1 |]) (fun () ->
              Nx.check P.unit ok () (fun i () -> Flag i)));
      test "data that does not broadcast to the flags raises" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; true |] in
          let data = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
          invalid ~by:"Nx.check" (fun () ->
              Nx.check P.tensor ok data (fun _ _ -> Failure "checked")));
    ]

let elsewhere =
  group "on another set"
    [
      test "a failure raises at the next wait on the flags' set" (fun () ->
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 3 |] [| true; false; false |])
          in
          Nx.check P.unit ok () (fun i () -> Flag i);
          raises (Flag [| 1 |]) (fun () -> Nx.sync ok);
          Nx.sync ok);
      test "flags of every set compute where the data has bytes" (fun () ->
          let ok = Nx.full Nx.bool [| 2 |] false in
          let data =
            Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 5.; 6. |])
          in
          Nx.check P.tensor ok data (fun i x -> At (i, Nx.item [] x, 0l, 0.));
          raises (At ([| 0 |], 5., 0l, 0.)) (fun () -> Nx.sync data));
      test "a failure found at a later check's issue raises at the next wait"
        (fun () ->
          let ok b = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; b |]) in
          let data =
            ( Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]),
              Nx.place S.on (Nx.create Nx.int32 [| 1 |] [| 9l |]) )
          in
          let found = ref 0 in
          let fail i (a, b) =
            incr found;
            At (i, Nx.item [] a, Nx.item [] b, 0.)
          in
          let desc = P.(pair tensor tensor) in
          Nx.check desc (ok false) data fail;
          Nx.check desc (ok true) data fail;
          equal ~msg:"found at the second issue" int 1 !found;
          raises (At ([| 1 |], 2., 9l, 0.)) (fun () -> Nx.sync (fst data));
          Nx.sync (fst data));
    ]

let walks =
  let key () = Nx.Rng.key 7 and keys () = P.v Nx.Rng.walk in
  let pass _ _ = Failure "the check passed its data" in
  group "walks that refuse 0-d arrays"
    [
      test "raise at the call on the host, naming Nx.check" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          invalid ~by:"Nx.check" (fun () -> Nx.check (keys ()) ok (key ()) pass));
      test "raise at the call on another set, leaving nothing pending"
        (fun () ->
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; false |])
          in
          invalid ~by:"Nx.check" (fun () -> Nx.check (keys ()) ok (key ()) pass);
          Nx.sync ok;
          Nx.check P.unit
            (Nx.place S.on (Nx.create Nx.bool [| 1 |] [| true |]))
            ()
            (fun i () -> Flag i);
          Nx.sync ok);
    ]

let () = exit (run "nx check" [ host; elsewhere; walks ])
