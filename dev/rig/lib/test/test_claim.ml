(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig
module B = Rig.Buffer
module Claim = Rig.Claim

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s =
  C.submit s ~reads ~writes ~waits

let timeout = 60.

(* Claims through views of one memory, against a model *)

type role = Read | Donate | Skip

(* A case: the memory's bytes, its views with what [with_] does with each,
   whether a reader claims the memory first, and whether [f] raises. *)
type case = {
  n : int;
  views : (int * int * role) list;
  claimed : bool;
  raises : bool;
}

let pp_case ppf c =
  let role = function Read -> "read" | Donate -> "donate" | Skip -> "-" in
  Format.fprintf ppf "%d bytes, claimed %b, raises %b, views %s" c.n c.claimed
    c.raises
    (String.concat " "
       (List.map
          (fun (o, l, r) -> Printf.sprintf "[%d,+%d %s]" o l (role r))
          c.views))

let case =
  let open Gen in
  let view n =
    bind (int_range 0 n) (fun o ->
        triple
          (constant ~pp:Format.pp_print_int o)
          (int_range 0 (n - o))
          (of_list
             ~pp:(fun ppf _ -> Format.pp_print_string ppf "role")
             [ Read; Donate; Skip ]))
  in
  let views n =
    bind
      (list ~size:(int_range 0 4) (view n))
      (fun vs ->
        (* The whole memory, donated, often enough to be alone. *)
        map (fun whole -> if whole then (0, n, Donate) :: vs else vs) bool)
  in
  with_pp pp_case
    (bind (int_range 1 32) (fun n ->
         map
           (fun (views, claimed, raises) -> { n; views; claimed; raises })
           (triple (views n) bool bool)))

let overlap (o, l, _) (o', l', _) = l > 0 && l' > 0 && o < o' + l' && o' < o + l

(* What [with_] must do: refuse when a donated view overlaps another claimed
   one; otherwise hold a donated view exclusive iff it spans the memory and
   nothing else claims it. *)
let expected c =
  let claimed = List.filter (fun (_, _, r) -> r <> Skip) c.views in
  let indexed = List.mapi (fun i v -> (i, v)) claimed in
  let others i = List.filter (fun (j, _) -> j <> i) indexed in
  let refused =
    List.exists
      (fun (i, ((_, _, r) as v)) ->
        r = Donate && List.exists (fun (_, w) -> overlap v w) (others i))
      indexed
  in
  if refused then None
  else
    Some
      (List.filter_map
         (fun (i, (o, l, r)) ->
           if r <> Donate then None
           else Some (o = 0 && l = c.n && (not c.claimed) && others i = []))
         indexed)

let law c =
  let b = B.create C.host c.n in
  let views =
    List.filter_map
      (fun (o, l, r) ->
        if r = Skip then None else Some (B.view b ~first:o ~length:l, r))
      c.views
  in
  let read =
    List.filter_map (fun (v, r) -> if r = Read then Some v else None) views
  in
  let donated =
    List.filter_map (fun (v, r) -> if r = Donate then Some v else None) views
  in
  if c.claimed then Claim.read b;
  let seen = ref [] in
  let got =
    match
      Claim.with_ ~read
        ~donate:(List.map (fun v -> [ v ]) donated)
        (fun cl ->
          seen := List.map (Claim.exclusive cl) donated;
          if c.raises then raise Exit)
    with
    | () | (exception Exit) -> Some !seen
    | exception Invalid_argument _ -> None
  in
  let e = expected c in
  cover "a donation refused" (e = None);
  cover "a donation held exclusive"
    (Option.fold ~none:false ~some:(List.mem true) e);
  cover "a donation of all the memory held for reading"
    (List.exists (fun (o, l, r) -> r = Donate && o = 0 && l = c.n) c.views
    && Option.fold ~none:false ~some:(fun ex -> not (List.mem true ex)) e);
  cover "a claim of no bytes"
    (List.exists (fun (_, l, r) -> l = 0 && r <> Skip) c.views);
  equal (option (list bool)) e got;
  (* Every claim [with_] took is released: only the first reader is left. *)
  if c.claimed then Claim.release b;
  raises_match Exn.invalid_arg (fun () -> Claim.release b);
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun cl ->
      equal ~msg:"alone again" bool true (Claim.exclusive cl b))

(* Cases *)

(* A donation overlapping a read that begins before a shorter read: the overlap
   is not with the read next to it in memory order. *)
let test_nested_overlap () =
  let b = B.create C.host 27 in
  let view first length = B.view b ~first ~length in
  raises_match Exn.invalid_arg (fun () ->
      Claim.with_ ~read:[ view 0 20; view 2 1 ] ~donate:[ [ view 19 1 ] ] ignore)

(* Memories of one device that has no address of them share no byte: two io
   memories are donated together, and views of one that overlap are refused. *)
let test_io_spans () =
  let io = Rig_support.machine "claim:io" in
  let a = B.create io 64 and b = B.create io 64 in
  Claim.with_ ~read:[] ~donate:[ [ a ]; [ b ] ] (fun cl ->
      equal ~msg:"both" (pair bool bool) (true, true)
        (Claim.exclusive cl a, Claim.exclusive cl b));
  raises_match ~msg:"views of one" Exn.invalid_arg (fun () ->
      Claim.with_
        ~read:[ B.view a ~first:0 ~length:32 ]
        ~donate:[ [ B.view a ~first:16 ~length:32 ] ]
        ignore)

let test_claims () =
  let b = B.create C.host 8 in
  Claim.read b;
  Claim.release b;
  raises_match Exn.invalid_arg (fun () -> Claim.release b);
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool true (Claim.exclusive c b);
      raises_match Exn.invalid_arg (fun () -> Claim.read b);
      raises_match Exn.invalid_arg (fun () ->
          Claim.with_ ~read:[ b ] ~donate:[] ignore))

(* A borrow and the memory it maps share one count. *)
let test_borrow_counts () =
  let d =
    require_ok ~pp:Format.pp_print_string (C.memory_device "claim:borrow")
  in
  let b = B.create C.host (1 lsl 16) in
  let borrowed = require_some (B.borrow d b) in
  Claim.read borrowed;
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool false (Claim.exclusive c b));
  Claim.release borrowed

let test_of_bigarray () =
  let b =
    B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8)
  in
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool false (Claim.exclusive c b))

(* Releasing a bigarray's memory that holds no read claim raises: the claim its
   holder keeps outside the claims stays, and the memory is never exclusive. *)
let test_release_kept () =
  let b =
    B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8)
  in
  raises_match Exn.invalid_arg (fun () -> Claim.release b);
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool false (Claim.exclusive c b))

(* A value of several buffers is exclusive only if each of them is. *)
let test_shards () =
  let d =
    require_ok ~pp:Format.pp_print_string (C.memory_device "claim:shards")
  in
  let a = B.create C.host 8 and b = B.create d 8 in
  Claim.with_ ~read:[]
    ~donate:[ [ a; b ] ]
    (fun c -> equal bool true (Claim.exclusive c a && Claim.exclusive c b));
  Claim.read b;
  Claim.with_ ~read:[]
    ~donate:[ [ a; b ] ]
    (fun c ->
      equal (pair bool bool) (false, false)
        (Claim.exclusive c a, Claim.exclusive c b));
  Claim.release b

(* Consumption *)

let dead why = Exn.invalid_arg ~substring:why

(* Memory consumed twice lives while the last buffer over it is reachable, and
   every earlier buffer is dead. *)
let test_consume_twice () =
  let b = B.create C.host (1 lsl 16) in
  let view = B.view b ~first:0 ~length:8 in
  let b' =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        Claim.consume c ~why:"first" b)
  in
  let b'' =
    Claim.with_ ~read:[] ~donate:[ [ b' ] ] (fun c ->
        Claim.consume c ~why:"second" b')
  in
  raises_match Exn.invalid_arg (fun () -> B.wait b B.Read);
  raises_match Exn.invalid_arg (fun () -> B.wait view B.Read);
  raises_match (dead "second") (fun () -> B.wait b' B.Read);
  Bigarray.Array1.fill (B.bigarray Bigarray.char b'') 'c';
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create C.host (1 lsl 16)));
  equal char 'c' (B.bigarray Bigarray.char b'').{(1 lsl 16) - 1}

let test_consume_refusals () =
  let b = B.create C.host 16 in
  let part = B.view b ~first:0 ~length:8 in
  Claim.with_ ~read:[] ~donate:[ [ part ] ] (fun c ->
      raises_match Exn.invalid_arg (fun () -> Claim.consume c ~why:"part" part));
  let other = B.create C.host 16 in
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      raises_match Exn.invalid_arg (fun () ->
          Claim.consume c ~why:"unclaimed" other))

(* A consumed memory's dead buffers refuse claims, and a release accepts
   them. *)
let test_dead_claims () =
  let b = B.create C.host 8 in
  Claim.read b;
  let live =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        Claim.consume c ~why:"donated" b)
  in
  Claim.release b;
  raises_match (dead "donated") (fun () -> Claim.read b);
  Claim.read live;
  Claim.release live

(* Two domains: with_ of a group of two memories, against a read of the second
   that ends in the same call. Whatever the order, no claim outlives its call:
   at the end the group is exclusive again. *)
type group = { b1 : B.t; b2 : B.t }

let release_group g =
  Claim.with_ ~read:[]
    ~donate:[ [ g.b1; g.b2 ] ]
    (fun c ->
      equal ~msg:"exclusive at the end" (pair bool bool) (true, true)
        (Claim.exclusive c g.b1, Claim.exclusive c g.b2))

let two = abstract ~release:release_group "g"
let make_group () = { b1 = B.create C.host 64; b2 = B.create C.host 64 }

let with_group g =
  Claim.with_ ~read:[]
    ~donate:[ [ g.b1; g.b2 ] ]
    (fun c -> Claim.exclusive c g.b1)

let read_group g =
  Claim.read g.b2;
  Claim.release g.b2

(* A call during another domain's with_ may find the memory held exclusive, and
   a with_ during a read finds the group read. *)
let judge_any () = function
  | Ok _ | Error (Invalid_argument _) -> ()
  | Error e -> raise e

let group_commands =
  [
    command "make" (Gen.unit @-> makes two) ignore make_group;
    command "with_" (two ^-> judges bool) judge_any with_group;
    command "read" (two ^-> judges unit) judge_any read_group;
  ]

(* Claims on memory whose stamps name a lost device raise Lost, and with_
   releases what it took first. *)
let test_lost_claims () =
  let module P = Rig_support.Polled in
  let d, p = P.open_ "claim:lost" in
  let m = B.create d 64 in
  let w = C.Submission.make ~reads:0 ~writes:1 d [||] in
  ignore (submit w ~writes:[| m |]);
  P.fail p;
  (try ignore (submit (C.Submission.make ~reads:0 ~writes:0 d [||]))
   with C.Lost _ -> ());
  let lost = function C.Lost _ -> true | _ -> false in
  raises_match lost (fun () -> Claim.read m);
  let h = B.create C.host 64 in
  raises_match lost (fun () -> Claim.with_ ~read:[ h; m ] ~donate:[] ignore);
  Claim.with_ ~read:[] ~donate:[ [ h ] ] (fun c ->
      equal ~msg:"h's claims released" bool true (Claim.exclusive c h))

(* A buffer's death is a fact with its reason: none while it lives, the
   consumption's reason once its memory is consumed; the buffer consume gives
   lives. *)
let test_dead_fact () =
  let b = B.create C.host 16 in
  equal ~msg:"live" (option string) None (B.dead b);
  let b' =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        Claim.consume c ~why:"donated" b)
  in
  equal ~msg:"consumed" (option string) (Some "donated") (B.dead b);
  equal ~msg:"its successor" (option string) None (B.dead b')

let tests =
  [
    group ~timeout "claims"
      [
        prop
          ~examples:
            [
              {
                n = 27;
                views =
                  [ (0, 0, Read); (19, 1, Donate); (2, 1, Read); (0, 20, Read) ];
                claimed = false;
                raises = false;
              };
            ]
          "with_ refuses overlapping donations and holds what spans alone" case
          law;
        test
          "a donation inside a long read with a short read between is refused"
          test_nested_overlap;
        test "memories with no address are claimed apart" test_io_spans;
        test "readers share a memory, an exclusive claim excludes them"
          test_claims;
        test "a borrow and the memory it maps share one count"
          test_borrow_counts;
        test "a bigarray's memory is never exclusive" test_of_bigarray;
        test "a release without a read claim keeps a bigarray's hidden one"
          test_release_kept;
        test "a value of several buffers is exclusive only if each is"
          test_shards;
      ];
    group ~timeout "consumption"
      [
        test "memory consumed twice lives while its last buffer does"
          test_consume_twice;
        test "only a claimed buffer that spans its memory is consumed"
          test_consume_refusals;
        test "a dead buffer refuses claims and accepts a release"
          test_dead_claims;
        test "a buffer's death is a fact with its reason" test_dead_fact;
        test "claims on memory a loss reaches raise and release"
          test_lost_claims;
      ];
    group ~timeout "domains"
      [
        stateful ~domains:2
          "with_ of a group and reads of its memory leave no claim behind"
          group_commands;
      ];
  ]

let () = exit (run "rig.claim" tests)
