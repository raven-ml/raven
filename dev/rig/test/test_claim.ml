(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Claim = Rig.Claim
module P = Rig_support.Polled
module R = Rig_support.Reader

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s =
  Rig.submit s ~run:(Rig.Submission.Run.make ()) ~reads ~writes ~waits

let timeout = 60.
let answer = Testable.make ~pp:R.pp_answer ~equal:( = )

(* Claims through views of one memory, against a model *)

type role = Read | Donate | Skip

(* How the memory is put outside the claims first: not at all, by an array
   exported over it, or by a share through a view of its last byte. *)
type outside = Inside | Exported | Shared

(* A case: the memory's bytes, its views with what [with_] does with each,
   whether a reader claims the memory first, how it is put outside the claims
   first, and whether [f] raises. *)
type case = {
  n : int;
  views : (int * int * role) list;
  claimed : bool;
  outside : outside;
  raises : bool;
}

let pp_case ppf c =
  let role = function Read -> "read" | Donate -> "donate" | Skip -> "-" in
  let outside = function
    | Inside -> "inside"
    | Exported -> "exported"
    | Shared -> "shared"
  in
  Format.fprintf ppf "%d bytes, claimed %b, %s, raises %b, views %s" c.n
    c.claimed (outside c.outside) c.raises
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
  let outside =
    of_list
      ~pp:(fun ppf _ -> Format.pp_print_string ppf "outside")
      [ Inside; Exported; Shared ]
  in
  let any n =
    map
      (fun (views, (claimed, outside), raises) ->
        { n; views; claimed; outside; raises })
      (triple (views n) (pair bool outside) bool)
  in
  (* The whole memory donated with no other claim, the one case held
     exclusive, drawn by construction as one case in four: [any] reaches it in
     one of about thirty. *)
  let alone n =
    map
      (fun raises ->
        {
          n;
          views = [ (0, n, Donate) ];
          claimed = false;
          outside = Inside;
          raises;
        })
      bool
  in
  with_pp pp_case
    (bind (int_range 1 32) (fun n ->
         bind (int_range 0 3) (fun k -> if k = 0 then alone n else any n)))

let overlap (o, l, _) (o', l', _) = l > 0 && l' > 0 && o < o' + l' && o' < o + l

(* What [with_] must do: refuse when a donated view overlaps another claimed
   one; otherwise hold a donated view exclusive iff it spans the memory, nothing
   else claims it and it is not outside the claims. *)
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
           else
             Some
               (o = 0 && l = c.n && (not c.claimed) && c.outside = Inside
               && others i = []))
         indexed)

let law c =
  let b = B.create Rig.host c.n in
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
  (match c.outside with
  | Inside -> ()
  | Exported -> ignore (B.bigarray Bigarray.char b)
  | Shared -> Claim.share (B.view b ~first:(c.n - 1) ~length:1));
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
  let alone = List.exists (fun (o, l, r) -> r = Donate && o = 0 && l = c.n) in
  cover "an exported memory alone"
    (c.outside = Exported && (not c.claimed) && alone c.views);
  cover "a shared memory alone"
    (c.outside = Shared && (not c.claimed) && alone c.views);
  equal (option (list bool)) e got;
  (* Every claim [with_] took is released: only the first reader is left. *)
  if c.claimed then Claim.release b;
  raises_match Exn.invalid_arg (fun () -> Claim.release b);
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun cl ->
      equal ~msg:"alone again" bool (c.outside = Inside) (Claim.exclusive cl b))

(* Cases *)

(* A donation overlapping a read that begins before a shorter read: the overlap
   is not with the read next to it in memory order. *)
let test_nested_overlap () =
  let b = B.create Rig.host 27 in
  let view first length = B.view b ~first ~length in
  raises_match Exn.invalid_arg (fun () ->
      Claim.with_ ~read:[ view 0 20; view 2 1 ] ~donate:[ [ view 19 1 ] ] ignore)

(* Memories of one device that has no address of them share no byte: two io
   memories are donated together, and views of one that overlap are refused. *)
let test_io_spans () =
  let io = Rig_support.io "claim:io" in
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
  let b = B.create Rig.host 8 in
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
    require_ok ~pp:Format.pp_print_string (Rig.memory_device "claim:borrow")
  in
  let b = B.create Rig.host (1 lsl 16) in
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

(* A share through a view puts all the memory outside the claims for good:
   no later with_ holds it exclusive, and readers claim and release it as
   before. *)
let test_share () =
  let b = B.create Rig.host 16 in
  Claim.share (B.view b ~first:8 ~length:4);
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal ~msg:"through a view" bool false (Claim.exclusive c b));
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal ~msg:"for good" bool false (Claim.exclusive c b));
  Claim.read b;
  Claim.release b;
  raises_match ~msg:"no read claim left" Exn.invalid_arg (fun () ->
      Claim.release b)

(* A share through a borrow puts the memory it maps outside the claims. *)
let test_share_borrow () =
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.memory_device "claim:share-borrow")
  in
  let b = B.create Rig.host (1 lsl 16) in
  Claim.share (require_some (B.borrow d b));
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool false (Claim.exclusive c b))

(* A value of several buffers is exclusive only if each of them is. *)
let test_shards () =
  let d =
    require_ok ~pp:Format.pp_print_string (Rig.memory_device "claim:shards")
  in
  let a = B.create Rig.host 8 and b = B.create d 8 in
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
  let b = B.create Rig.host (1 lsl 16) in
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
  ignore (Sys.opaque_identity (B.create Rig.host (1 lsl 16)));
  equal char 'c' (B.bigarray Bigarray.char b'').{(1 lsl 16) - 1}

let test_consume_refusals () =
  let b = B.create Rig.host 16 in
  let part = B.view b ~first:0 ~length:8 in
  Claim.with_ ~read:[] ~donate:[ [ part ] ] (fun c ->
      raises_match Exn.invalid_arg (fun () -> Claim.consume c ~why:"part" part));
  let other = B.create Rig.host 16 in
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      raises_match Exn.invalid_arg (fun () ->
          Claim.consume c ~why:"unclaimed" other))

(* A consumed memory's dead buffers refuse claims, and a release accepts
   them. *)
let test_dead_claims () =
  let b = B.create Rig.host 8 in
  Claim.read b;
  let live =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        Claim.consume c ~why:"donated" b)
  in
  Claim.release b;
  raises_match (dead "donated") (fun () -> Claim.read b);
  Claim.read live;
  Claim.release live

(* A share under an exclusive claim is refused and marks nothing; once the
   memory is consumed the dead buffer is refused, and the consumption's buffer
   shares. *)
let test_share_refusals () =
  let b = B.create Rig.host 16 in
  let b' =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        raises_match ~msg:"held exclusive" Exn.invalid_arg (fun () ->
            Claim.share b);
        Claim.consume c ~why:"donated" b)
  in
  raises_match ~msg:"dead" (dead "donated") (fun () -> Claim.share b);
  Claim.with_ ~read:[] ~donate:[ [ b' ] ] (fun c ->
      equal ~msg:"the refusal marked nothing" bool true (Claim.exclusive c b'));
  Claim.share b';
  Claim.with_ ~read:[] ~donate:[ [ b' ] ] (fun c ->
      equal ~msg:"shared" bool false (Claim.exclusive c b'))

(* A consumer that shares or exports the buffer consume gave holds it exclusive
   no longer. *)
let test_share_consumed () =
  let shared = B.create Rig.host 16 in
  Claim.with_ ~read:[] ~donate:[ [ shared ] ] (fun c ->
      let b = Claim.consume c ~why:"donated" shared in
      equal ~msg:"consumed" bool true (Claim.exclusive c b);
      Claim.share b;
      equal ~msg:"shared" bool false (Claim.exclusive c b));
  let exported = B.create Rig.host 16 in
  Claim.with_ ~read:[] ~donate:[ [ exported ] ] (fun c ->
      let b = Claim.consume c ~why:"donated" exported in
      ignore (B.bigarray Bigarray.char b);
      equal ~msg:"exported" bool false (Claim.exclusive c b))

(* Claims that outlive their with_ hold nothing: beside a live with_ over the
   same memory, a stale claim is not exclusive and consumes nothing. *)
let test_stale_returned () =
  let b = B.create Rig.host 64 in
  let stale = Claim.with_ ~read:[] ~donate:[ [ b ] ] Fun.id in
  equal ~msg:"alone" bool false (Claim.exclusive stale b);
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal ~msg:"the live claim" bool true (Claim.exclusive c b);
      equal ~msg:"beside a live claim" bool false (Claim.exclusive stale b);
      raises_match Exn.invalid_arg (fun () ->
          Claim.consume stale ~why:"stale" b);
      ignore (Claim.consume c ~why:"live" b))

(* A with_ whose [f] raised ends its claims too: beside a reader, the stale
   claim is not exclusive and consumes nothing. *)
let test_stale_raised () =
  let b = B.create Rig.host 64 in
  let stale = ref None in
  raises (Failure "f") (fun () ->
      Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
          stale := Some c;
          failwith "f"));
  let stale = require_some !stale in
  Claim.read b;
  equal bool false (Claim.exclusive stale b);
  raises_match Exn.invalid_arg (fun () -> Claim.consume stale ~why:"stale" b);
  Claim.release b;
  equal ~msg:"the buffer stays live" (option string) None (B.dead b)

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
let make_group () = { b1 = B.create Rig.host 64; b2 = B.create Rig.host 64 }

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

(* Two domains: reads, exports and shares against a donation that consumes the
   memory when it holds it exclusive, and writes it in place through the buffer
   the consumption gives. A read, an export or a share before the donation
   keeps it a read; one after a consumption finds the buffer dead. A donation
   may also stay a read while another domain's claims hold the memory. *)
type memory = { mutable dead : bool; mutable reads : int; mutable out : bool }

let memory =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "dead %b, reads %d, outside %b" r.dead r.reads r.out)
    "m"

let judge_read r = function
  | Ok () ->
      equal ~msg:"dead" bool false r.dead;
      r.reads <- r.reads + 1
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.dead
  | Error e -> raise e

let judge_outside r = function
  | Ok () ->
      equal ~msg:"dead" bool false r.dead;
      r.out <- true
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.dead
  | Error e -> raise e

let judge_donate r = function
  | Ok consumed ->
      equal ~msg:"dead" bool false r.dead;
      if consumed then
        equal ~msg:"reads, outside" (pair int bool) (0, false) (r.reads, r.out);
      r.dead <- consumed
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.dead
  | Error e -> raise e

let judge_c_claim r = function
  | Ok R.Claimed ->
      equal ~msg:"dead" bool false r.dead;
      r.reads <- r.reads + 1
  | Ok (R.Dead | R.Exclusive) -> equal ~msg:"dead" bool true r.dead
  | Ok a -> failf "claim answered %a" R.pp_answer a
  | Error e -> raise e

let export b = ignore (B.bigarray Bigarray.char b)

let donate b =
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      Claim.exclusive c b
      && begin
        let b' = Claim.consume c ~why:"donated" b in
        Bigarray.Array1.fill (B.bigarray Bigarray.char b') 'd';
        true
      end)

let donation_commands =
  [
    command "make"
      (Gen.unit @-> makes memory)
      (fun () -> { dead = false; reads = 0; out = false })
      (fun () -> B.create Rig.host 64);
    command "read" (memory ^-> judges unit) judge_read Claim.read;
    command "claim from C"
      (memory ^-> judges answer)
      judge_c_claim
      (fun b -> R.claim b B.Read);
    command "export" (memory ^-> judges unit) judge_outside export;
    command "share" (memory ^-> judges unit) judge_outside Claim.share;
    command "donate" (memory ^-> judges bool) judge_donate donate;
  ]

(* An export under an exclusive claim is the consumer's: refused before the
   consumption, made after it, and the memory is outside the claims once they
   are released. *)
let test_export_exclusive () =
  let b = B.create Rig.host 16 in
  let b' =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        equal ~msg:"exclusive" bool true (Claim.exclusive c b);
        raises_match ~msg:"before the consumption" Exn.invalid_arg (fun () ->
            B.bigarray Bigarray.char b);
        let b' = Claim.consume c ~why:"donated" b in
        Bigarray.Array1.fill (B.bigarray Bigarray.char b') 'x';
        b')
  in
  Claim.with_ ~read:[] ~donate:[ [ b' ] ] (fun c ->
      equal ~msg:"after the release" bool false (Claim.exclusive c b'));
  raises_match ~msg:"no read claim left" Exn.invalid_arg (fun () ->
      Claim.release b')

(* Claims from C *)

(* A claim from C is a read claim: it sits beside other readers, keeps a
   donation a read until it is released, and is refused under an exclusive claim
   and on a dead buffer, where it leaves no claim behind. *)
let test_c_claims () =
  let b = B.create Rig.host 16 in
  equal ~msg:"for reading" answer R.Claimed (R.claim b B.Read);
  equal ~msg:"for writing" answer R.Claimed (R.claim b B.Read_write);
  Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal ~msg:"donated under two claims" bool false (Claim.exclusive c b));
  R.release b;
  R.release b;
  let b' =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        equal ~msg:"donated once released" bool true (Claim.exclusive c b);
        equal ~msg:"under an exclusive claim" answer R.Exclusive
          (R.claim b B.Read);
        Claim.consume c ~why:"donated" b)
  in
  equal ~msg:"dead" answer R.Dead (R.claim b B.Read);
  equal ~msg:"its reason" (option string) (Some "donated") (R.why b);
  Claim.with_ ~read:[] ~donate:[ [ b' ] ] (fun c ->
      equal ~msg:"no claim left" bool true (Claim.exclusive c b'))

let write d m =
  ignore
    (submit (Rig.Submission.make ~reads:0 ~writes:1 d [||]) ~writes:[| m |])

let read d m =
  ignore (submit (Rig.Submission.make ~reads:1 ~writes:0 d [||]) ~reads:[| m |])

(* Whether a claim holds [m]'s memory: a donation of it stays a read. *)
let claimed m =
  Claim.with_ ~read:[] ~donate:[ [ m ] ] (fun c -> not (Claim.exclusive c m))

(* A claim waits for nothing: it answers Wait while a point the access must
   follow is unreached, the last write for reading and every use for writing,
   and holds the claim then; Claimed once the word shows the point. A wait
   under the claim returns once the points are reached. *)
let test_c_wait () =
  let d, p = P.open_ "claim:c-wait" in
  let m = B.create d 64 in
  write d m;
  equal ~msg:"an unreached write" answer R.Wait (R.claim m B.Read);
  equal ~msg:"claimed while it waits" bool true (claimed m);
  R.wait m B.Read;
  equal ~msg:"the wait ran the write" int 0 (P.queued p);
  R.release m;
  equal ~msg:"the write reached" answer R.Claimed (R.claim m B.Read);
  R.release m;
  read d m;
  equal ~msg:"an unreached read, for reading" answer R.Claimed
    (R.claim m B.Read);
  R.release m;
  equal ~msg:"an unreached read, for writing" answer R.Wait
    (R.claim m B.Read_write);
  R.wait m B.Read_write;
  R.release m;
  equal ~msg:"after a wait" answer R.Claimed (R.claim m B.Read_write);
  R.release m;
  equal ~msg:"no claim left" bool false (claimed m)

(* Behind a transport a claim reads the value the host last read: work that ran
   answers Wait until a wait reads the word, and Claimed after it. *)
let test_c_transport () =
  let d, p = P.open_ ~transport:true "claim:c-transport" in
  let m = B.create d 64 in
  write d m;
  ignore (P.run p);
  equal ~msg:"ran, unread" answer R.Wait (R.claim m B.Read);
  R.wait m B.Read;
  R.release m;
  equal ~msg:"after a wait" answer R.Claimed (R.claim m B.Read);
  R.release m

let lost = function Rig.Lost _ -> true | _ -> false

(* A lost device's memory answers Wait even once its work on it was reached:
   the wait under the claim raises Lost and leaves the claim held. *)
let test_c_lost () =
  let d, p = P.open_ "claim:c-lost" in
  let m = B.create d 64 in
  write d m;
  ignore (P.run p);
  P.fail p;
  (try ignore (submit (Rig.Submission.make ~reads:0 ~writes:0 d [||]))
   with Rig.Lost _ -> ());
  equal ~msg:"reached, lost" answer R.Wait (R.claim m B.Read);
  raises_match lost (fun () -> R.wait m B.Read);
  Claim.release m;
  raises_match ~msg:"the one claim released"
    (Exn.invalid_arg ~substring:"no read claim")
    (fun () -> Claim.release m)

(* A device lost during the wait: the wait raises Lost, the claim held. *)
let test_c_lost_waiting () =
  let d, p = P.open_ "claim:c-lost-waiting" in
  let m = B.create d 64 in
  write d m;
  equal ~msg:"unreached" answer R.Wait (R.claim m B.Read);
  P.fault p "gone";
  raises_match lost (fun () -> R.wait m B.Read);
  Claim.release m;
  raises_match ~msg:"the one claim released"
    (Exn.invalid_arg ~substring:"no read claim")
    (fun () -> Claim.release m)

(* Other memory answers Wait for a point a lost device did not reach, and
   Claimed for one it reached: its stop's last value in the word reaches
   nothing. *)
let test_c_lost_points () =
  let d, p = P.open_ "claim:c-lost-points" in
  let reached = B.create Rig.host (1 lsl 16) in
  let unreached = B.create Rig.host (1 lsl 16) in
  write d (require_some (B.borrow d reached));
  ignore (P.run p);
  write d (require_some (B.borrow d unreached));
  P.fail p;
  (try ignore (submit (Rig.Submission.make ~reads:0 ~writes:0 d [||]))
   with Rig.Lost _ -> ());
  equal ~msg:"reached" answer R.Claimed (R.claim reached B.Read_write);
  R.release reached;
  equal ~msg:"unreached" answer R.Wait (R.claim unreached B.Read);
  R.release unreached

(* Fixed memory is ordered by its access: a claim to read waits for an
   unreached submission that writes it, and for none that reads it. *)
let test_c_fixed () =
  let d, p = P.open_ "claim:c-fixed" in
  let m = B.create d 64 in
  let fixed access = Rig.Submission.make ~fixed:[ (m, access) ] ~reads:0 ~writes:0 d [||] in
  ignore (submit (fixed B.Read));
  equal ~msg:"an unreached read" answer R.Claimed (R.claim m B.Read);
  R.release m;
  ignore (submit (fixed B.Read_write));
  equal ~msg:"an unreached write" answer R.Wait (R.claim m B.Read);
  R.release m;
  ignore (P.run p);
  equal ~msg:"reached" answer R.Claimed (R.claim m B.Read);
  R.release m

(* Claims on a lost device's memory raise Lost, and with_ releases what it took
   first. *)
let test_lost_claims () =
  let d, p = P.open_ "claim:lost" in
  let m = B.create d 64 in
  let w = Rig.Submission.make ~reads:0 ~writes:1 d [||] in
  ignore (submit w ~writes:[| m |]);
  P.fail p;
  (try ignore (submit (Rig.Submission.make ~reads:0 ~writes:0 d [||]))
   with Rig.Lost _ -> ());
  raises_match lost (fun () -> Claim.read m);
  let h = B.create Rig.host 64 in
  raises_match lost (fun () -> Claim.with_ ~read:[ h; m ] ~donate:[] ignore);
  Claim.with_ ~read:[] ~donate:[ [ h ] ] (fun c ->
      equal ~msg:"h's claims released" bool true (Claim.exclusive c h))

(* A buffer's death is a fact with its reason: none while it lives, the
   consumption's reason once its memory is consumed; the buffer consume gives
   lives. *)
let test_dead_fact () =
  let b = B.create Rig.host 16 in
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
                outside = Inside;
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
        test "a share through a view is never exclusive again" test_share;
        test "a share through a borrow reaches the memory it maps"
          test_share_borrow;
        test "a value of several buffers is exclusive only if each is"
          test_shards;
        test "an export under an exclusive claim waits for the consumption"
          test_export_exclusive;
      ];
    group ~timeout "consumption"
      [
        test "memory consumed twice lives while its last buffer does"
          test_consume_twice;
        test "only a claimed buffer that spans its memory is consumed"
          test_consume_refusals;
        test "a dead buffer refuses claims and accepts a release"
          test_dead_claims;
        test "a share waits for the consumption and refuses its dead buffer"
          test_share_refusals;
        test "a consumer that shares its buffer holds it exclusive no longer"
          test_share_consumed;
        test "a claim whose with_ returned holds nothing" test_stale_returned;
        test "a claim whose with_ raised holds nothing" test_stale_raised;
        test "a buffer's death is a fact with its reason" test_dead_fact;
        test "claims on memory a loss reaches raise and release"
          test_lost_claims;
      ];
    group ~timeout "claims from C"
      [
        test "a claim from C is a read claim" test_c_claims;
        test
          "a claim from C holds the memory and waits until the word shows \
           its points"
          test_c_wait;
        test "behind a transport a claim reads the word a wait read"
          test_c_transport;
        test "a wait on a lost device's memory raises Lost under the claim"
          test_c_lost;
        test "a device lost during a wait from C raises Lost under the claim"
          test_c_lost_waiting;
        test "a claim on memory a lost device did not reach waits"
          test_c_lost_points;
        test "a claim on fixed memory follows its uses by their access"
          test_c_fixed;
      ];
    group ~timeout "domains"
      [
        stateful ~domains:2
          "with_ of a group and reads of its memory leave no claim behind"
          group_commands;
        stateful ~domains:2
          "reads, exports and shares beside a consuming donation are ordered \
           with it"
          donation_commands;
      ];
  ]

let () = exit (run "rig.claim" tests)
