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
module S2 = (val Nx.devices [ m 0; m 1 ])
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ m 1 ])

exception At of int array * float * int32 * float
exception Flag of int array
exception Element of string

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let numel s = Array.fold_left ( * ) 1 s

(* Placements over any one set. *)
let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
let pass _ _ = Failure "the check passed its data"

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
      test "a failure's 0-d values lie where their data lies" (fun () ->
          let ok = Nx.full_at S.on Nx.bool [| 2 |] false in
          let data =
            Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 5.; 6. |])
          in
          let lies = ref None in
          Nx.check P.tensor ok data (fun i x ->
              lies := Some (Nx.placement x);
              At (i, Nx.item [] x, 0l, 0.));
          raises (At ([| 0 |], 5., 0l, 0.)) (fun () -> Nx.sync data);
          equal ~msg:"the 0-d value's placement"
            (option (placement ()))
            (Some S.on) !lies);
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

(* One array, whose walk passes the array it is given through [f]. *)
let through f = P.v (fun c x -> f (P.Walk.tensor c x))

let tried =
  group "the rebuild tried at the call"
    [
      test "is over 0-d stand-ins at each array's placement, uncut" (fun () ->
          let at = S2.split ~axis:0 in
          let ok = Nx.full_at at Nx.bool [| 2 |] true in
          let x = Nx.place at (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
          let seen = ref [] in
          let record y =
            seen := (Nx.shape y, Nx.placement y) :: !seen;
            y
          in
          Nx.check (through record) ok x pass;
          Nx.sync ok;
          match !seen with
          | [] -> fail "the walk saw no array"
          | (shape, lies) :: _ ->
              equal ~msg:"the stand-in's shape" (array int) [||] shape;
              equal ~msg:"the stand-in's placement" (placement ()) S2.on lies);
      test "computes nothing on the devices" (fun () ->
          let ok = Nx.full_at Count.on Nx.bool [| 4 |] true in
          let data =
            ( Nx.full_at Count.on Nx.float32 [| 4 |] 1.,
              Nx.full_at Count.on Nx.int32 [| 4 |] 2l )
          in
          let calls f =
            C.reset ();
            f ();
            Nx.sync ok;
            C.calls ()
          in
          let bare = calls (fun () -> Nx.check P.unit ok () pass) in
          let full =
            calls (fun () -> Nx.check P.(pair tensor tensor) ok data pass)
          in
          equal ~msg:"kernel calls with data, as without" int bare full);
      test "raises where the walk computes on a stand-in" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
          raises
            (Invalid_argument
               "Nx.check: the structure's walk refuses 0-d arrays: Nx.add: a \
                value traced by Nx.check reached eager execution") (fun () ->
              Nx.check (through (fun y -> Nx.add y y)) ok x pass));
    ]

let walks =
  let key () = Nx.Rng.key 7 and keys () = P.v Nx.Rng.walk in
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
          let key = Nx.Rng.place S.on (key ()) in
          invalid ~by:"Nx.check" (fun () -> Nx.check (keys ()) ok key pass);
          Nx.sync ok;
          Nx.check P.unit
            (Nx.place S.on (Nx.create Nx.bool [| 1 |] [| true |]))
            ()
            (fun i () -> Flag i);
          Nx.sync ok);
    ]

let () = exit (run "nx check" [ host; elsewhere; tried; walks ])
