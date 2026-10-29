(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Claim = Rig.Claim
module P = Rig_support.Polled
module R = Rig_support.Reader

(* A read claim taken and ended apart, as a library's C code takes one
   ([rig_buffer_claim] in [rig.h]). *)
let claim_read b =
  match R.claim b Rig.Read with
  | R.Claimed | Wait -> ()
  | a -> invalid_arg (Format.asprintf "a claim answered %a" R.pp_answer a)

let claim_release b = R.release b

(* A read claim, then the wait [a] follows under it, as [Buffer.read] takes
   them. *)
let claim_wait b a =
  claim_read b;
  match R.wait b a with
  | () -> ()
  | exception e ->
      claim_release b;
      raise e

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits (Rig.Submission.Run.make s) ~buffers

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
  let view roles n =
    bind (int_range 0 n) (fun o ->
        triple
          (constant ~pp:Format.pp_print_int o)
          (int_range 0 (n - o))
          (of_list
             ~pp:(fun ppf _ -> Format.pp_print_string ppf "role")
             roles))
  in
  let views size roles n =
    bind
      (list ~size (view roles n))
      (fun vs ->
        (* The whole memory, donated, often enough to be alone. *)
        map (fun whole -> if whole then (0, n, Donate) :: vs else vs) bool)
  in
  let outside =
    of_list
      ~pp:(fun ppf _ -> Format.pp_print_string ppf "outside")
      [ Inside; Exported; Shared ]
  in
  let any ?(size = int_range 0 24) ?(roles = [ Read; Donate; Skip ]) n =
    map
      (fun (views, (claimed, outside), raises) ->
        { n; views; claimed; outside; raises })
      (triple (views size roles n) (pair bool outside) bool)
  in
  (* Nine donated views or more, which [with_] checks by sorting them all,
     drawn as one case in four from nine views or more, three in five of them
     donated: [any] reaches them in about one case in eighteen. *)
  let many n =
    any ~size:(int_range 9 24) ~roles:[ Read; Donate; Donate; Donate; Skip ] n
  in
  (* The whole memory donated with no other view, drawn by construction as one
     case in four: [any] reaches it in one of about thirty. Half of these have
     no other claim, the one case held exclusive; the others are claimed or
     outside the claims, which holds the donation for reading. *)
  let alone n =
    map
      (fun ((claimed, outside), raises) ->
        { n; views = [ (0, n, Donate) ]; claimed; outside; raises })
      (pair
         (frequency [ (1, pair (constant false) (constant Inside));
                      (1, pair bool outside) ])
         bool)
  in
  with_pp pp_case
    (bind (int_range 1 32) (fun n ->
         bind (int_range 0 3) (fun k ->
             if k = 0 then alone n else if k = 1 then many n else any n)))

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
  if c.claimed then claim_read b;
  (match c.outside with
  | Inside -> ()
  | Exported -> ignore (B.export b Bigarray.char)
  | Shared -> Claim.share (B.view b ~first:(c.n - 1) ~length:1));
  let seen = ref [] in
  let got =
    match
      Claim.with_ ~read
        ~donate:donated
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
  cover "nine or more donated buffers"
    (List.length (List.filter (fun (_, _, r) -> r = Donate) c.views) >= 9);
  cover "a claim of no bytes"
    (List.exists (fun (_, l, r) -> l = 0 && r <> Skip) c.views);
  let alone = List.exists (fun (o, l, r) -> r = Donate && o = 0 && l = c.n) in
  cover "an exported memory alone"
    (c.outside = Exported && (not c.claimed) && alone c.views);
  cover "a shared memory alone"
    (c.outside = Shared && (not c.claimed) && alone c.views);
  equal (option (list bool)) e got;
  (* Every claim [with_] took is released: only the first reader is left. *)
  if c.claimed then claim_release b;

  Claim.with_ ~read:[] ~donate:[ b ] (fun cl ->
      equal ~msg:"alone again" bool (c.outside = Inside) (Claim.exclusive cl b))

(* Cases *)

(* Donations that overlap each other are refused, with nothing else
   claimed. *)
let test_donations_overlap () =
  let b = B.create Rig.host 16 in
  let view first length = B.view b ~first ~length in
  raises_match Exn.invalid_arg (fun () ->
      Claim.with_ ~read:[] ~donate:[ view 0 8; view 4 8 ] ignore);
  Claim.with_ ~read:[] ~donate:[ view 0 8; view 8 8 ] (fun c ->
      equal ~msg:"apart, neither spans" (pair bool bool) (false, false)
        (Claim.exclusive c (view 0 8), Claim.exclusive c (view 8 8)))

(* Memories of one device that has no address of them share no byte: two io
   memories are donated together, and views of one that overlap are refused. *)
let test_io_spans () =
  let io = Rig_support.io "claim:io" in
  let a = B.create io 64 and b = B.create io 64 in
  Claim.with_ ~read:[] ~donate:[ a; b ] (fun cl ->
      equal ~msg:"both" (pair bool bool) (true, true)
        (Claim.exclusive cl a, Claim.exclusive cl b));
  raises_match ~msg:"views of one" Exn.invalid_arg (fun () ->
      Claim.with_
        ~read:[ B.view a ~first:0 ~length:32 ]
        ~donate:[ B.view a ~first:16 ~length:32 ]
        ignore)

let test_claims () =
  let b = B.create Rig.host 8 in
  claim_read b;
  claim_release b;

  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      equal bool true (Claim.exclusive c b);
      raises_match Exn.invalid_arg (fun () ->
          Claim.with_ ~read:[ b ] ~donate:[] ignore))

(* A donor uses memory it holds exclusive only through the buffer that
   consumes it. Before the consumption, the host's reads and writes in place
   and every copy, blit or fill, whether the host, a device or staging memory
   runs it, are refused. After it, each of them runs on the consumed
   buffer. *)
let test_exclusive_writes () =
  let d, _ = P.open_ ~host_visible:false "claim:exclusive-writes" in
  let staging, _ =
    P.open_ ~maps_host:false ~host_visible:false "claim:exclusive-staged"
  in
  let src = B.of_string "abcdefgh" in
  let written b =
    let s = Bytes.create 8 in
    B.blit_to_bytes b 0 s 0 8;
    Bytes.to_string s
  in
  let refused msg f =
    raises_match ~msg (Exn.invalid_arg ~substring:"held exclusive") f
  in
  let unconsumed b =
    refused "a read" (fun () -> B.read b Bigarray.char ignore);
    refused "a write" (fun () -> B.write b Bigarray.char ignore);
    refused "a copy into it" (fun () -> B.copy ~src ~dst:b);
    refused "a copy to the host" (fun () ->
        B.copy ~src:b ~dst:(B.create Rig.host 8));
    refused "a copy through staging memory" (fun () ->
        B.copy ~src:b ~dst:(B.create staging 8));
    refused "a blit into it" (fun () -> B.blit_from_string "ijkl" 0 b 4 4);
    refused "a blit out of it" (fun () ->
        B.blit_to_bytes b 0 (Bytes.create 8) 0 8);
    refused "a fill" (fun () -> B.fill b 'z')
  in
  let consumed b =
    B.copy ~src ~dst:b;
    B.blit_from_string "ijkl" 0 b 4 4;
    B.fill (B.view b ~first:0 ~length:2) 'z';
    let s = B.create staging 8 in
    B.copy ~src:b ~dst:s;
    B.copy ~src:s ~dst:b;
    b
  in
  let donated b =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        equal ~msg:"exclusive" bool true (Claim.exclusive c b);
        unconsumed b;
        consumed (Claim.consume c ~why:"donated" b))
  in
  let h = donated (B.create Rig.host 8) in
  equal string ~msg:"host memory" "zzcdijkl" (written h);
  let m = donated (B.create d 8) in
  equal string ~msg:"memory the host does not address" "zzcdijkl" (written m)

(* A borrow of memory a donation holds exclusive raises: it claims the
   memory while it maps it, so that no free returns the memory meanwhile. *)
let test_borrow_exclusive () =
  let d = Result.get_ok (Rig.memory_device "claim:borrow-exclusive") in
  let b = B.create Rig.host (1 lsl 16) in
  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      equal ~msg:"exclusive" bool true (Claim.exclusive c b);
      raises_match (Exn.invalid_arg ~substring:"held exclusive") (fun () ->
          B.borrow d b));
  is_some ~msg:"once the donation returned" (B.borrow d b)

(* A borrow and the memory it maps share one count. *)
let test_borrow_counts () =
  let d = Result.get_ok (Rig.memory_device "claim:borrow") in
  let b = B.create Rig.host (1 lsl 16) in
  let borrowed = require_some (B.borrow d b) in
  claim_read borrowed;
  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      equal bool false (Claim.exclusive c b));
  claim_release borrowed

let test_of_bigarray () =
  let b =
    B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8)
  in
  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      equal bool false (Claim.exclusive c b))

(* Releasing a bigarray's memory that holds no read claim raises: the claim its
   holder keeps outside the claims stays, and the memory is never exclusive. *)
let donated_exclusive b =
  Claim.with_ ~read:[] ~donate:[ b ] (fun c -> Claim.exclusive c b)

(* Memory shared through one buffer is never exclusive through another: a view,
   a borrow, the buffer itself. Sharing it again changes nothing, and readers
   claim and release it as before. *)
let test_share_covers () =
  let d = Result.get_ok (Rig.memory_device "claim:share") in
  let fresh () =
    let b = B.create Rig.host (1 lsl 16) in
    (b, B.view b ~first:0 ~length:8, require_some (B.borrow d b))
  in
  let b, view, borrowed = fresh () in
  equal ~msg:"unshared" bool true (donated_exclusive b);
  Claim.share view;
  equal ~msg:"shared through a view, the buffer" bool false
    (donated_exclusive b);
  equal ~msg:"shared through a view, the borrow" bool false
    (donated_exclusive borrowed);
  Claim.share view;
  Claim.share b;
  equal ~msg:"shared again" bool false (donated_exclusive b);
  claim_read b;
  claim_release b;

  let b, _, borrowed = fresh () in
  Claim.share borrowed;
  equal ~msg:"shared through the borrow" bool false (donated_exclusive b);
  let b, _, borrowed = fresh () in
  Claim.share b;
  equal ~msg:"shared through the buffer" bool false (donated_exclusive borrowed)

(* Each buffer of a donation is held exclusive on its own: a value of several
   buffers is donated when each of them is. *)
let test_shards () =
  let d = Result.get_ok (Rig.memory_device "claim:shards") in
  let a = B.create Rig.host 8 and b = B.create d 8 in
  Claim.with_ ~read:[]
    ~donate:[ a; b ]
    (fun c -> equal bool true (Claim.exclusive c a && Claim.exclusive c b));
  claim_read b;
  Claim.with_ ~read:[]
    ~donate:[ a; b ]
    (fun c ->
      equal (pair bool bool) (true, false)
        (Claim.exclusive c a, Claim.exclusive c b));
  claim_release b

(* Consumption *)

let dead why = Exn.invalid_arg ~substring:why

(* Memory consumed twice lives while the last buffer over it is reachable, and
   every earlier buffer is dead. *)
let test_consume_twice () =
  let b = B.create Rig.host (1 lsl 16) in
  let view = B.view b ~first:0 ~length:8 in
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        Claim.consume c ~why:"first" b)
  in
  let b'' =
    Claim.with_ ~read:[] ~donate:[ b' ] (fun c ->
        Claim.consume c ~why:"second" b')
  in
  raises_match Exn.invalid_arg (fun () -> B.wait b Rig.Read);
  raises_match Exn.invalid_arg (fun () -> B.wait view Rig.Read);
  raises_match (dead "second") (fun () -> B.wait b' Rig.Read);
  B.write b'' Bigarray.char (fun a -> Bigarray.Array1.fill a 'c');
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create Rig.host (1 lsl 16)));
  equal char 'c' (B.read b'' Bigarray.char (fun a -> a.{(1 lsl 16) - 1}))

(* A consumption of memory that another claim reads kills every earlier buffer
   as any other, and the buffer it gives admits only reads, through its views
   too, so the reader's bytes are never written in place. Consumed again where
   the donation holds it exclusive, the memory admits writes again. *)
let test_consume_read () =
  let access =
    Testable.make
      ~pp:(fun ppf a ->
        Format.pp_print_string ppf
          (match a with Rig.Read -> "Read" | Rig.Read_write -> "Read_write"))
      ~equal:( = )
  in
  let reads msg f =
    raises_match ~msg (Exn.invalid_arg ~substring:"admits only reads") f
  in
  let b = B.create Rig.host 8 in
  B.fill b 'a';
  let b' =
    Claim.with_ ~read:[ b ] ~donate:[] (fun _ ->
        Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
            equal ~msg:"exclusive" bool false (Claim.exclusive c b);
            let b' = Claim.consume c ~why:"donated" b in
            reads "a write under the reader's claim" (fun () ->
                B.write b' Bigarray.char ignore);
            b'))
  in
  equal ~msg:"the earlier buffer" (option string) (Some "donated") (B.dead b);
  equal ~msg:"the consumed buffer's access" access Rig.Read (B.access b');
  equal ~msg:"a view's access" access Rig.Read
    (B.access (B.view b' ~first:0 ~length:4));
  reads "a write" (fun () -> B.write b' Bigarray.char ignore);
  reads "a fill" (fun () -> B.fill b' 'z');
  reads "a copy into it" (fun () -> B.copy ~src:(B.of_string "12345678") ~dst:b');
  equal ~msg:"a claim from C to write" answer R.Read_only
    (R.claim b' Rig.Read_write);
  equal ~msg:"its bytes" char 'a' (B.read b' Bigarray.char (fun a -> a.{7}));
  let b'' =
    Claim.with_ ~read:[] ~donate:[ b' ] (fun c ->
        equal ~msg:"exclusive again" bool true (Claim.exclusive c b');
        Claim.consume c ~why:"again" b')
  in
  equal ~msg:"consumed exclusive" access Rig.Read_write (B.access b'');
  B.fill b'' 'z';
  equal ~msg:"written" char 'z' (B.read b'' Bigarray.char (fun a -> a.{7}))

let test_consume_refusals () =
  let b = B.create Rig.host 16 in
  let part = B.view b ~first:0 ~length:8 in
  Claim.with_ ~read:[] ~donate:[ part ] (fun c ->
      raises_match Exn.invalid_arg (fun () -> Claim.consume c ~why:"part" part));
  let other = B.create Rig.host 16 in
  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      raises_match Exn.invalid_arg (fun () ->
          Claim.consume c ~why:"unclaimed" other))

(* A consumed memory's dead buffers refuse claims, and a release accepts
   them. *)
let test_dead_claims () =
  let b = B.create Rig.host 8 in
  claim_read b;
  let live =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        Claim.consume c ~why:"donated" b)
  in
  claim_release b;
  raises_match (dead "donated") (fun () -> Claim.with_ ~read:[ b ] ~donate:[] ignore);
  claim_read live;
  claim_release live

(* A buffer of a closed device answers that device's loss, also once a later
   device holds its slot: the C claim reads the slot as the device's only while
   the slot's line names it. The later device's first value shows that it took
   the slot. *)
let test_closed_slot_claim () =
  let d, _ = P.open_ "claim:closed" in
  let p =
    Rig.submit (Rig.Submission.Run.make (Rig.Submission.make d [||]))
      ~buffers:[||]
  in
  Rig.Point.wait p;
  let b = B.create d 0 in
  Rig.close d;
  Rig.free_cache d;
  let later, _ = P.open_ "claim:later" in
  equal ~msg:"the later device took the closed one's slot" int
    (Rig.Point.value p) (Rig.submitted later);
  equal ~msg:"the C claim" answer R.Lost (R.claim b Rig.Read);
  Rig.close later

(* Sharing a dead buffer, or memory claims hold exclusive, raises and changes
   nothing. The buffer a consumption gives is shared as any other. *)
let test_share_refused () =
  let b = B.create Rig.host 16 in
  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      raises_match ~msg:"held" Exn.invalid_arg (fun () -> Claim.share b);
      equal ~msg:"held, after the refusal" bool true (Claim.exclusive c b));
  equal ~msg:"unshared" bool true (donated_exclusive b);
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        Claim.consume c ~why:"donated" b)
  in
  raises_match ~msg:"dead" (dead "donated") (fun () -> Claim.share b);
  equal ~msg:"the live buffer" bool true (donated_exclusive b');
  Claim.share b';
  equal ~msg:"the live buffer, shared" bool false (donated_exclusive b');
  raises_match ~msg:"dead, its memory shared" (dead "donated") (fun () ->
      Claim.share b);
  let b = B.create Rig.host 16 in
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        let b' = Claim.consume c ~why:"consumed" b in
        Claim.share b';
        b')
  in
  equal ~msg:"shared while consumed" bool false (donated_exclusive b')

(* Claims that outlive their with_ hold nothing: beside a live with_ over the
   same memory, a stale claim is not exclusive and consumes nothing. *)
let test_stale_returned () =
  let b = B.create Rig.host 64 in
  let stale = Claim.with_ ~read:[] ~donate:[ b ] Fun.id in
  equal ~msg:"alone" bool false (Claim.exclusive stale b);
  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
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
      Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
          stale := Some c;
          failwith "f"));
  let stale = require_some !stale in
  claim_read b;
  equal bool false (Claim.exclusive stale b);
  raises_match Exn.invalid_arg (fun () -> Claim.consume stale ~why:"stale" b);
  claim_release b;
  equal ~msg:"the buffer stays live" (option string) None (B.dead b)

(* What a donation of [b] alone finds: its memory exclusive, shared by another
   claim, or the refusal of a with_. *)
let donation b =
  match donated_exclusive b with
  | true -> "exclusive"
  | false -> "shared"
  | exception Invalid_argument m -> m

(* A signal pending as [f] returns or raises lands after with_ has ended its
   claims. [f]'s last act raises SIGINT, which the runtime records; its
   handler raises [Exit] at the next poll point, which the allocation after
   the call places if with_ has none left. *)
exception Raised

let interrupted_at_end f =
  let r = B.create Rig.host 8 and b = B.create Rig.host 8 in
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      raises Exit (fun () ->
          (match Claim.with_ ~read:[ r ] ~donate:[ b ] f with
          | () -> ()
          | exception Raised -> ());
          ignore (Sys.opaque_identity (ref ()))));
  equal ~msg:"read, donated" (pair string string) ("exclusive", "exclusive")
    (donation r, donation b)

let test_signal_returned () =
  interrupted_at_end (fun _ -> Rig_support.interrupt ())

let test_signal_raised () =
  interrupted_at_end (fun _ ->
      Rig_support.interrupt ();
      raise_notrace Raised)

(* Claims of several buffers from C *)

(* What a buffer of a case is to [rig_buffers_claim]: live, read or written;
   with a write it must wait for; dead; held exclusive; or admitting only
   reads, claimed to write. *)
type state = Live of Rig.access | Pending | Dead | Held | Read_only

let state =
  Gen.of_list
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | Live Rig.Read -> "read"
        | Live Rig.Read_write -> "written"
        | Pending -> "pending"
        | Dead -> "dead"
        | Held -> "held"
        | Read_only -> "read-only"))
    [ Live Rig.Read; Live Rig.Read_write; Pending; Dead; Held; Read_only ]

let states = Gen.list ~size:(Gen.int_range 0 6) state

let refusal = function
  | Dead -> Some R.Dead
  | Held -> Some R.Exclusive
  | Read_only -> Some R.Read_only
  | Live _ | Pending -> None

let claims_device = lazy (P.open_ "claim:claim-all")

(* A buffer in [s], the access it is claimed for, and whether its memory
   should donate once the claims end. *)
let buffer_in s =
  let d, _ = Lazy.force claims_device in
  let consumed b = Rig_support.donated b |> Option.get in
  match s with
  | Live a -> (B.create Rig.host 8, a)
  | Pending ->
      let b = B.create d 8 in
      let s =
        Rig.Submission.make d
          [| Rig_support.touching [| Rig.Submission.write |] |]
      in
      ignore (submit s ~buffers:[| b |]);
      (b, Rig.Read)
  | Dead ->
      let b = B.create Rig.host 8 in
      ignore (consumed b);
      (b, Rig.Read)
  | Held -> (B.create Rig.host 8, Rig.Read)
  | Read_only ->
      let b = B.create Rig.host 8 in
      let b' =
        Claim.with_ ~read:[ b ] ~donate:[] (fun _ ->
            Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
                Claim.consume c ~why:"consumed" b))
      in
      (b', Rig.Read_write)

(* [rig_buffers_claim] claims every buffer or none: it refuses at the first
   buffer that [rig_buffer_claim] would refuse, holding nothing, and otherwise
   answers each buffer as [rig_buffer_claim] would and holds them all until
   [rig_buffers_release]. *)
let claim_all_law ss =
  let bs = List.map buffer_in ss in
  let buffers = Array.of_list (List.map fst bs) in
  let access = Array.of_list (List.map snd bs) in
  let held =
    List.filter_map (fun (s, (b, _)) -> if s = Held then Some b else None)
      (List.combine ss bs)
  in
  let free () =
    List.iter2
      (fun s (b, _) ->
        match s with
        | Live _ | Pending ->
            equal ~msg:"no claim left" bool true (donated_exclusive b)
        | Dead | Held | Read_only -> ())
      ss bs
  in
  let expected =
    List.find_mapi
      (fun i s -> Option.map (fun a -> (i, a)) (refusal s))
      ss
  in
  let outcome = Claim.with_ ~read:[] ~donate:held (fun _ ->
      R.claim_all buffers access)
  in
  (match (outcome, expected) with
  | R.Refused (i, a), Some (i', a') ->
      cover "a refusal after a claim" (i > 0);
      equal ~msg:"the refused buffer" (pair int answer) (i', a') (i, a)
  | R.All answers, None ->
      cover "every buffer of several claimed" (Array.length answers > 1);
      equal ~msg:"the answers" (array answer)
        (Array.of_list
           (List.map (function Pending -> R.Wait | _ -> R.Claimed) ss))
        answers;
      List.iter2
        (fun s (b, _) ->
          if s <> Held then
            equal ~msg:"claimed" bool false (donated_exclusive b))
        ss bs;
      R.release_all buffers
  | R.All _, Some _ -> fail "every buffer claimed past a refusal"
  | R.Refused (i, a), None ->
      failf "buffer %d refused (%a) with none to refuse" i R.pp_answer a);
  free ();
  ignore (P.run (snd (Lazy.force claims_device)))

(* Holds *)

(* After the holds over a memory are unreachable, a call that moves collected
   memory releases them: a donation then finds the memory exclusive. *)
let collect_holds () =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create Rig.host 8))

(* A hold keeps its memory live through donations, which stay read claims and
   consume nothing, while reads and writes go on. Once the hold is collected,
   the first call that moves collected memory, on any device, releases it. *)
let test_hold () =
  let d, _ = P.open_ "claim:hold" in
  let a = B.create Rig.host 64 and b = B.create d 64 in
  let h = Claim.hold [ a; b ] in
  Claim.with_ ~read:[] ~donate:[ a; b ] (fun c ->
      equal ~msg:"exclusive" (pair bool bool) (false, false)
        (Claim.exclusive c a, Claim.exclusive c b);
      equal ~msg:"consumable" (pair bool bool) (false, false)
        (Claim.consumable c a, Claim.consumable c b);
      raises_match ~msg:"a consumption" (Exn.invalid_arg ~substring:"held")
        (fun () -> Claim.consume c ~why:"donated" a));
  equal ~msg:"live" (option string) None (B.dead a);
  B.fill a 'h';
  equal ~msg:"written under the hold" char 'h'
    (B.read a Bigarray.char (fun x -> x.{0}));
  ignore (Sys.opaque_identity h);
  collect_holds ();
  Claim.with_ ~read:[] ~donate:[ a; b ] (fun c ->
      equal ~msg:"exclusive once collected" (pair bool bool) (true, true)
        (Claim.exclusive c a, Claim.exclusive c b))

(* A release ends a hold at once: while the hold is still reachable and
   nothing was collected, a donation finds its memory exclusive and consumes
   it. *)
let test_release () =
  let d, _ = P.open_ "claim:release" in
  let a = B.create Rig.host 64 and b = B.create d 64 in
  let h = Claim.hold [ a; b ] in
  Claim.release h;
  Claim.with_ ~read:[] ~donate:[ a; b ] (fun c ->
      equal ~msg:"exclusive" (pair bool bool) (true, true)
        (Claim.exclusive c a, Claim.exclusive c b);
      ignore (Claim.consume c ~why:"donated" a));
  equal ~msg:"consumed" (option string) (Some "donated") (B.dead a);
  ignore (Sys.opaque_identity h)

(* A later release of a hold, and its collection, end nothing: another hold of
   the same memory holds it until its own release. *)
let test_release_once () =
  let a = B.create Rig.host 64 in
  let h1 = Claim.hold [ a ] and h2 = Claim.hold [ a ] in
  Claim.release h1;
  Claim.release h1;
  equal ~msg:"after a second release" bool false (donated_exclusive a);
  ignore (Sys.opaque_identity h1);
  collect_holds ();
  equal ~msg:"after the collection" bool false (donated_exclusive a);
  Claim.release h2;
  equal ~msg:"after the other's release" bool true (donated_exclusive a);
  Claim.release (Claim.hold []);
  ignore (Sys.opaque_identity h2)

(* Two domains: releases of one hold over two memories, beside drops of the
   hold, which release it, collections and donations of both memories. Another
   hold keeps the second memory to the end. A release after the first ends
   nothing, so the first memory is exclusive once the hold is released and
   the second never is. A drop beside a release releases the hold twice on two
   domains, and a collection after it runs the drain of the hold's record
   beside the other domain's release. *)
type releases = { mutable released : bool }

type held_pair = {
  first : B.t;
  second : B.t;
  hold : Claim.hold option Atomic.t;
  keeper : Claim.hold;
}

let held_pair =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "released %b" r.released)
    ~release:(fun p -> ignore (Sys.opaque_identity p.keeper))
    "p"

let make_pair () =
  let first = B.create Rig.host 64 and second = B.create Rig.host 64 in
  let hold = Atomic.make (Some (Claim.hold [ first; second ])) in
  { first; second; hold; keeper = Claim.hold [ second ] }

let release_pair p = Option.iter Claim.release (Atomic.get p.hold)
let drop_pair p = Option.iter Claim.release (Atomic.exchange p.hold None)

(* A collection that drains: a minor collection finalises a token dropped
   while young, and the allocation drains the list of holds, on which a
   release or the finaliser put the hold's record. *)
let collect_pair (_ : held_pair) =
  Gc.minor ();
  ignore (Sys.opaque_identity (B.create Rig.host 8))

let donate_pair p =
  Claim.with_ ~read:[]
    ~donate:[ p.first; p.second ]
    (fun c -> (Claim.exclusive c p.first, Claim.exclusive c p.second))

let release_model r = r.released <- true

(* The kept memory is never exclusive, and the first only once the hold is
   released. Two donations side by side may each find the other's read claim
   on the first and get a read claim, or one holds it exclusive and the other
   raises. *)
let judge_donation r = function
  | Ok (first, second) ->
      equal ~msg:"the kept memory exclusive" bool false second;
      if not r.released then equal ~msg:"held, exclusive" bool false first
  | Error (Invalid_argument _) -> equal ~msg:"released" bool true r.released
  | Error e -> raise e

let release_commands =
  [
    command "make" (Gen.unit @-> makes held_pair)
      (fun () -> { released = false })
      make_pair;
    command "release"
      (held_pair ^-> returns unit)
      release_model release_pair;
    command "drop"
      (held_pair ^-> returns unit)
      release_model drop_pair;
    command "collect" (held_pair ^-> returns unit) ignore collect_pair;
    command "donate"
      (held_pair ^-> judges (pair bool bool))
      judge_donation donate_pair;
  ]

(* A submission holds its fixed memory and the buffers its parts name, which
   every run reads, as a hold does: none of them donates while the submission
   is reachable, and each does once it is collected. Slots hold nothing. *)
let test_submission_holds () =
  let d, _ = P.open_ "claim:submission-holds" in
  let fixed = B.create d 8 and src = B.create d 8 and dst = B.create d 8 in
  let ws = B.create Rig.host 8 and slot = B.create d 8 in
  let all = [ fixed; src; dst; ws; slot ] in
  let exclusive () =
    Claim.with_ ~read:[] ~donate:all (fun c ->
        List.map (Claim.exclusive c) all)
  in
  let s =
    Rig.Submission.(
      make
        ~fixed:[ (fixed, Rig.Read) ]
        d
        [|
          copy ~src:(Buffer src) ~dst:(Buffer dst);
          words ws;
          copy ~src:(Slot 0) ~dst:(Buffer dst);
        |])
  in
  equal ~msg:"while the submission is reachable" (list bool)
    [ false; false; false; false; true ]
    (exclusive ());
  ignore (Sys.opaque_identity (submit s ~buffers:[| slot |]));
  collect_holds ();
  equal ~msg:"once it is collected" (list bool)
    [ true; true; true; true; true ]
    (exclusive ())

(* A hold of a dead buffer, or of memory held exclusive, raises and holds none
   of the buffers it was given. *)
let test_hold_refusals () =
  let a = B.create Rig.host 8 and b = B.create Rig.host 8 in
  Claim.with_ ~read:[] ~donate:[ b ] (fun _ ->
      raises_match ~msg:"held exclusive"
        (Exn.invalid_arg ~substring:"held exclusive")
        (fun () -> Claim.hold [ a; b ]));
  ignore (require_some (Rig_support.donated b));
  raises_match ~msg:"dead" (dead "donated") (fun () -> Claim.hold [ a; b ]);
  equal ~msg:"the other buffer" bool true (donated_exclusive a)

(* [consumable c b] says whether [consume c b] succeeds: for memory [c]
   claims, live, spanned by [b] and not held, read by another claim or not. *)
let test_consumable () =
  let b = B.create Rig.host 16 and other = B.create Rig.host 16 in
  let part = B.view b ~first:0 ~length:8 in
  Claim.with_ ~read:[ b ] ~donate:[] (fun c ->
      equal ~msg:"a view" bool false (Claim.consumable c part);
      equal ~msg:"unclaimed" bool false (Claim.consumable c other);
      claim_read b;
      equal ~msg:"read by another claim" bool true (Claim.consumable c b);
      claim_release b;
      ignore (Claim.consume c ~why:"consumed" b);
      equal ~msg:"dead" bool false (Claim.consumable c b));
  let stale = Claim.with_ ~read:[ other ] ~donate:[] Fun.id in
  equal ~msg:"ended" bool false (Claim.consumable stale other)

(* Two domains: holds of a memory against consumptions of it under a read
   claim. A consumption succeeds only while no hold holds the memory, and kills
   it; a hold of a consumed memory raises. Whole calls interleave here: the
   window between a consumption's check for holds and its raise of the
   generation lies inside one C call with no poll point, which no test can
   hold, so its order is argued in claim.mli. *)
type held = { mutable consumed : bool; mutable holds : int }

let held_memory =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "consumed %b, holds %d" r.consumed r.holds)
    "m"

(* The holds taken, kept reachable to the end of the run. *)
let kept : Claim.hold list Atomic.t = Atomic.make []

let rec keep h =
  let hs = Atomic.get kept in
  if not (Atomic.compare_and_set kept hs (h :: hs)) then keep h

let take_hold b = keep (Claim.hold [ b ])

let judge_hold r = function
  | Ok () ->
      equal ~msg:"consumed" bool false r.consumed;
      r.holds <- r.holds + 1
  | Error (Invalid_argument _) -> equal ~msg:"consumed" bool true r.consumed
  | Error e -> raise e

(* What a consumption under a read claim found: the memory consumed, held, or
   the buffer dead. A held memory is never consumed later, as the holds are
   kept, so a refusal of a buffer still live is a hold's. *)
let consume_read b =
  match
    Claim.with_ ~read:[ b ] ~donate:[] (fun c ->
        ignore (Claim.consume c ~why:"consumed" b))
  with
  | () -> "consumed"
  | exception Invalid_argument _ -> if B.dead b = None then "held" else "dead"

let judge_consume r = function
  | Ok "consumed" ->
      equal ~msg:"consumed, held" (pair bool int) (false, 0)
        (r.consumed, r.holds);
      r.consumed <- true
  | Ok "held" ->
      equal ~msg:"consumed" bool false r.consumed;
      greater ~msg:"holds" int ~than:0 r.holds
  | Ok "dead" -> equal ~msg:"consumed" bool true r.consumed
  | Ok what -> failf "a consumption found %s" what
  | Error e -> raise e

let hold_commands =
  [
    command "make"
      (Gen.unit @-> makes held_memory)
      (fun () -> { consumed = false; holds = 0 })
      (fun () -> B.create Rig.host 64);
    command "hold" (held_memory ^-> judges unit) judge_hold take_hold;
    command "consume"
      (held_memory ^-> judges string)
      judge_consume consume_read;
  ]

(* Two domains: with_ of a group of two memories, against a read of the second
   that ends in the same call. Whatever the order, no claim outlives its call:
   at the end the group is exclusive again. *)
type group = { b1 : B.t; b2 : B.t }

let release_group g =
  Claim.with_ ~read:[]
    ~donate:[ g.b1; g.b2 ]
    (fun c ->
      equal ~msg:"exclusive at the end" (pair bool bool) (true, true)
        (Claim.exclusive c g.b1, Claim.exclusive c g.b2))

let two = abstract ~release:release_group "g"
let make_group () = { b1 = B.create Rig.host 64; b2 = B.create Rig.host 64 }

let with_group g =
  Claim.with_ ~read:[]
    ~donate:[ g.b1; g.b2 ]
    (fun c -> Claim.exclusive c g.b1)

let read_group g =
  claim_read g.b2;
  claim_release g.b2

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
   the consumption gives, or a free, which consumes it as a donation does. A
   read, an export or a share before the donation keeps it a read; one after a
   consumption finds the buffer dead. A donation may also stay a read while
   another domain's claims hold the memory. *)
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

(* A free consumes as a donation does. One that keeps the memory found
   another claim on it, such as another domain's donation holding it
   exclusive before it consumes, which no order of whole calls shows. *)
let judge_free r = function
  | Ok true ->
      equal ~msg:"dead, reads, outside" (triple bool int bool)
        (false, 0, false) (r.dead, r.reads, r.out);
      r.dead <- true
  | Ok false -> ()
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.dead
  | Error e -> raise e

let judge_c_claim r = function
  | Ok R.Claimed ->
      equal ~msg:"dead" bool false r.dead;
      r.reads <- r.reads + 1
  | Ok (R.Dead | R.Exclusive) -> equal ~msg:"dead" bool true r.dead
  | Ok a -> failf "claim answered %a" R.pp_answer a
  | Error e -> raise e

let export b = ignore (B.export b Bigarray.char)

let donate b =
  match Rig_support.donated b with
  | None -> false
  | Some b' ->
      B.write b' Bigarray.char (fun a -> Bigarray.Array1.fill a 'd');
      true

let donation_commands =
  [
    command "make"
      (Gen.unit @-> makes memory)
      (fun () -> { dead = false; reads = 0; out = false })
      (fun () -> B.create Rig.host 64);
    command "read" (memory ^-> judges unit) judge_read claim_read;
    command "claim from C"
      (memory ^-> judges answer)
      judge_c_claim
      (fun b -> R.claim b Rig.Read);
    command "export" (memory ^-> judges unit) judge_outside export;
    command "share" (memory ^-> judges unit) judge_outside Claim.share;
    command "donate" (memory ^-> judges bool) judge_donate donate;
    command "free" (memory ^-> judges bool) judge_free B.free;
  ]

(* Two domains: a donor that hands its claims to another domain and returns at
   once, consumptions through those claims there, and donations of the same
   memory. A consumption through the handed claims succeeds only before their
   with_ has returned, and raises after. A donation's own consumption never
   finds its buffer dead, and the memory is consumed at most once. A with_
   beside another's may find the memory held exclusive or read, which no order
   of whole calls shows, so those refusals and reads are accepted in any
   state. So is a hand-over that finds claims handed: another domain's
   hand-over puts them in the cell before its with_ returns, and a
   consumption through them may follow. *)
type handed = { x : B.t; cell : Claim.t option Atomic.t }
type handing = { mutable handed : bool; mutable live : bool }

let handing =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "handed %b, live %b" r.handed r.live)
    "h"

let hand h =
  if Option.is_some (Atomic.get h.cell) then "already"
  else
    match
      Claim.with_ ~read:[] ~donate:[ h.x ] (fun c ->
          Atomic.compare_and_set h.cell None (Some c))
    with
    | true -> "handed"
    | false -> "already"
    | exception Invalid_argument _ -> "refused"

let consume_handed h =
  match Atomic.get h.cell with
  | None -> "none"
  | Some c -> (
      match Claim.consume c ~why:"handed" h.x with
      | _ -> "consumed"
      | exception Invalid_argument _ -> "refused")

let donate_handed h =
  let own c =
    if not (Claim.exclusive c h.x) then "shared"
    else
      match Claim.consume c ~why:"donated" h.x with
      | _ -> "consumed"
      | exception Invalid_argument _ -> "own consumption refused"
  in
  match Claim.with_ ~read:[] ~donate:[ h.x ] own with
  | s -> s
  | exception Invalid_argument _ -> "refused"

let judge_hand r = function
  | Ok "handed" ->
      equal ~msg:"handed before" bool false r.handed;
      r.handed <- true
  | Ok ("already" | "refused") -> ()
  | Ok s -> failf "the hand-over answered %s" s
  | Error e -> raise e

let judge_handed_consume r = function
  | Ok "none" -> equal ~msg:"handed" bool false r.handed
  | Ok "consumed" ->
      equal ~msg:"handed, live" (pair bool bool) (false, true)
        (r.handed, r.live);
      r.live <- false
  | Ok "refused" -> equal ~msg:"handed or dead" bool true (r.handed || not r.live)
  | Ok s -> failf "the consumption answered %s" s
  | Error e -> raise e

let judge_handed_donate r = function
  | Ok "consumed" ->
      equal ~msg:"live" bool true r.live;
      r.live <- false
  | Ok "shared" -> equal ~msg:"live" bool true r.live
  | Ok "refused" -> ()
  | Ok s -> failf "the donation answered %s" s
  | Error e -> raise e

let handed_commands =
  [
    command "make"
      (Gen.unit @-> makes handing)
      (fun () -> { handed = false; live = true })
      (fun () -> { x = B.create Rig.host 64; cell = Atomic.make None });
    command "hand" (handing ^-> judges string) judge_hand hand;
    command "consume handed"
      (handing ^-> judges string)
      judge_handed_consume consume_handed;
    command "donate" (handing ^-> judges string) judge_handed_donate
      donate_handed;
  ]

(* An export under an exclusive claim is the consumer's: refused before the
   consumption, made after it, and the memory is outside the claims once they
   are released. *)
let test_export_exclusive () =
  let b = B.create Rig.host 16 in
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        equal ~msg:"exclusive" bool true (Claim.exclusive c b);
        raises_match ~msg:"before the consumption" Exn.invalid_arg (fun () ->
            B.export b Bigarray.char);
        let b' = Claim.consume c ~why:"donated" b in
        Bigarray.Array1.fill (B.export b' Bigarray.char) 'x';
        b')
  in
  Claim.with_ ~read:[] ~donate:[ b' ] (fun c ->
      equal ~msg:"after the release" bool false (Claim.exclusive c b'))

(* Claims from C *)

(* A claim from C is a read claim: it sits beside other readers, keeps a
   donation a read until it is released, and is refused under an exclusive claim
   and on a dead buffer, where it leaves no claim behind. The buffer a
   consumption gives is claimed at once. *)
let test_c_claims () =
  let b = B.create Rig.host 16 in
  equal ~msg:"for reading" answer R.Claimed (R.claim b Rig.Read);
  equal ~msg:"for writing" answer R.Claimed (R.claim b Rig.Read_write);
  Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      equal ~msg:"donated under two claims" bool false (Claim.exclusive c b));
  R.release b;
  R.release b;
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        equal ~msg:"donated once released" bool true (Claim.exclusive c b);
        equal ~msg:"under an exclusive claim" answer R.Exclusive
          (R.claim b Rig.Read);
        let b' = Claim.consume c ~why:"donated" b in
        equal ~msg:"the consumed buffer" answer R.Claimed (R.claim b' Rig.Read);
        R.release b';
        b')
  in
  equal ~msg:"dead" answer R.Dead (R.claim b Rig.Read);
  equal ~msg:"its reason" (option string) (Some "donated") (R.why b);
  Claim.with_ ~read:[] ~donate:[ b' ] (fun c ->
      equal ~msg:"no claim left" bool true (Claim.exclusive c b'))

let write d m =
  let s = Rig.Submission.make d [| Rig_support.touching [| Rig.Submission.write |] |] in
  ignore (submit s ~buffers:[| m |])

let read d m =
  let s = Rig.Submission.make d [| Rig_support.touching [| Rig.Submission.read |] |] in
  ignore (submit s ~buffers:[| m |])

(* Whether a claim holds [m]'s memory: a donation of it stays a read. *)
let claimed m =
  Claim.with_ ~read:[] ~donate:[ m ] (fun c -> not (Claim.exclusive c m))

(* A claim waits for nothing: it answers Wait while a point the access must
   follow is unreached, the last write for reading and every use for writing,
   and holds the claim then; Claimed once the word shows the point. A wait
   under the claim returns once the points are reached. *)
let test_c_wait () =
  let d, p = P.open_ "claim:c-wait" in
  let m = B.create d 64 in
  write d m;
  equal ~msg:"an unreached write" answer R.Wait (R.claim m Rig.Read);
  equal ~msg:"claimed while it waits" bool true (claimed m);
  R.wait m Rig.Read;
  equal ~msg:"the wait ran the write" int 0 (P.queued p);
  R.release m;
  equal ~msg:"the write reached" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  read d m;
  equal ~msg:"an unreached read, for reading" answer R.Claimed
    (R.claim m Rig.Read);
  R.release m;
  equal ~msg:"an unreached read, for writing" answer R.Wait
    (R.claim m Rig.Read_write);
  R.wait m Rig.Read_write;
  R.release m;
  equal ~msg:"after a wait" answer R.Claimed (R.claim m Rig.Read_write);
  R.release m;
  equal ~msg:"no claim left" bool false (claimed m)

(* Behind a transport a claim reads the value the host last read: work that ran
   answers Wait until a wait reads the word, and Claimed after it. *)
let test_c_transport () =
  let d, p = P.open_ ~transport:true "claim:c-transport" in
  let m = B.create d 64 in
  write d m;
  ignore (P.run p);
  equal ~msg:"ran, unread" answer R.Wait (R.claim m Rig.Read);
  R.wait m Rig.Read;
  R.release m;
  equal ~msg:"after a wait" answer R.Claimed (R.claim m Rig.Read);
  R.release m

let lost = function Rig.Lost _ -> true | _ -> false
let lost_by d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* Loses [d]: its next submit fails. *)
let lose d p =
  P.fail p;
  try ignore (submit (Rig.Submission.make d [||]))
  with Rig.Lost _ -> ()

(* A lost device's memory answers Lost even once its work on it was reached,
   holding the claim: the wait under it raises Lost. *)
let test_c_lost () =
  let d, p = P.open_ "claim:c-lost" in
  let m = B.create d 64 in
  write d m;
  ignore (P.run p);
  lose d p;
  equal ~msg:"reached, lost" answer R.Lost (R.claim m Rig.Read);
  raises_match (lost_by d) (fun () -> R.wait m Rig.Read);
  claim_release m

(* A device lost during the wait: the wait raises Lost, the claim held. *)
let test_c_lost_waiting () =
  let d, p = P.open_ "claim:c-lost-waiting" in
  let m = B.create d 64 in
  write d m;
  equal ~msg:"unreached" answer R.Wait (R.claim m Rig.Read);
  P.fault p "gone";
  raises_match lost (fun () -> R.wait m Rig.Read);
  claim_release m

(* A wait from C holds its claim while waiting and once it returns. *)
let test_wait () =
  let d, p = P.open_ "claim:wait" in
  let m = B.create d 64 in
  write d m;
  P.gate p;
  let waiting = Domain.spawn (fun () -> claim_wait m Rig.Read) in
  Fun.protect
    ~finally:(fun () ->
      P.open_gate p;
      Domain.join waiting)
    (fun () ->
      Rig_support.await "a claim waiting at the gate" (fun () ->
          P.sleepers p = 1);
      Claim.with_ ~read:[] ~donate:[ m ] (fun c ->
          equal ~msg:"a donation while the wait is blocked" bool false
            (Claim.exclusive c m)));
  equal ~msg:"the write is done" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  Claim.with_ ~read:[] ~donate:[ m ] (fun c ->
      equal bool ~msg:"a donation under the claim" false (Claim.exclusive c m));
  claim_release m;
  Claim.with_ ~read:[] ~donate:[ m ] (fun c ->
      equal bool ~msg:"a donation once released" true (Claim.exclusive c m))

(* Hold submit at a foreign wait, after it has collected the source's uses. A
   donation must not overwrite the source before that submit publishes its own
   use, whether the submission names the source as fixed memory or a copy part
   names it. *)
let test_submit_donation (name, kind) () =
  let module Sub = Rig.Submission in
  let producer, pp = P.open_ ("claim:submit-producer:" ^ name) in
  let d, p = P.open_ ("claim:submit:" ^ name) in
  let src = B.create d 8 and dst = B.create d 8 in
  Rig_support.store (R.host src) 42;
  Rig_support.store (R.host dst) 0;
  let part work = work in
  let run, buffers =
    match kind with
    | `Copy ->
        ( Sub.Run.make
            (Sub.make d [| part Sub.(copy ~src:(Slot 0) ~dst:(Slot 1)) |]),
          [| src; dst |] )
    | `Fixed ->
        let fill =
          Sub.fill Rig_support.carry ~params:24
        in
        let s =
          Sub.make
            ~fixed:[ (src, Rig.Read); (dst, Rig.Read_write) ]
            d
            [| part fill |]
        in
        let run = Sub.Run.make s and block = Sub.block s 0 in
        Sub.Run.int64 run block 0 (Int64.of_int (Sub.address s dst));
        Sub.Run.int64 run block 8 (Int64.of_int (Sub.address s src));
        Sub.Run.int64 run block 16 8L;
        (* The submit runs on another domain. *)
        Sub.Run.release run;
        (run, [||])
  in
  (* A submission holds its fixed memory for its life, and claims a run's
     buffers only while a submit runs. *)
  let free = kind = `Copy in
  equal ~msg:"the source once the submission is made" bool free
    (donated_exclusive src);
  let point = submit (Sub.make producer [||]) in
  P.gate pp;
  let waiting =
    Domain.spawn (fun () -> Rig.submit run ~waits:[| point |] ~buffers)
  in
  let exclusive = ref false in
  Fun.protect
    ~finally:(fun () ->
      P.open_gate pp;
      Rig.Point.wait (Domain.join waiting))
    (fun () ->
      Rig_support.await "submit waiting at the producer's gate" (fun () ->
          P.sleepers pp = 1);
      Claim.with_ ~read:[] ~donate:[ src ] (fun c ->
          exclusive := Claim.exclusive c src;
          if !exclusive then begin
            let replacement = Claim.consume c ~why:"reused" src in
            B.wait replacement Rig.Read_write;
            Rig_support.store (R.host replacement) 99
          end));
  equal ~msg:"donation and the submitted copy's result" (pair bool int)
    (false, 42)
    (!exclusive, Rig_support.load (R.host dst));
  equal ~msg:"the submitted work completed" int 0 (P.queued p);
  equal ~msg:"the source after hand-over" bool free (donated_exclusive src)

(* A submit refused for a run's buffer releases every claim it took on the
   others, each buffer passed in two slots. *)
let test_run_claim_refusals () =
  let module Sub = Rig.Submission in
  let d, p = P.open_ "claim:run-refusals" in
  let buffers () = Array.init 5 (fun _ -> B.create d 8) in
  let s = Sub.make d [| Rig_support.touching (Array.make 10 Sub.read) |] in
  let slots bs = Array.init 10 (fun i -> bs.(i / 2)) in
  let check bs =
    Array.iteri
      (fun i b ->
        equal ~msg:(Printf.sprintf "buffer %d has no claim left" i) bool true
          (donated_exclusive b))
      bs
  in
  let bs = buffers () in
  Array.iter
    (fun b ->
      Claim.with_ ~read:[] ~donate:[ b ] (fun _ ->
          raises_match (Exn.invalid_arg ~substring:"held exclusive")
            (fun () -> submit s ~buffers:(slots bs)));
      check bs)
    bs;
  ignore (submit s ~buffers:(slots bs));
  check bs;
  for i = 0 to Array.length bs - 1 do
    let bs = buffers () in
    let replacement =
      Claim.with_ ~read:[] ~donate:[ bs.(i) ] (fun c ->
          Claim.consume c ~why:"replaced" bs.(i))
    in
    raises_match (Exn.invalid_arg ~substring:"replaced") (fun () ->
        submit s ~buffers:(slots bs));
    bs.(i) <- replacement;
    check bs
  done;
  ignore (P.run p)

(* A submit whose wait raises releases the claims on its run's buffers. *)
let test_run_wait_raises () =
  let module Sub = Rig.Submission in
  let producer, pp = P.open_ "claim:run-raises-producer" in
  let d, p = P.open_ "claim:run-raises" in
  let a = B.create d 8 and b = B.create d 8 in
  let s = Sub.make d [| Rig_support.touching (Array.make 3 Sub.read) |] in
  let buffers = [| a; b; a |] in
  let run = Sub.Run.make s in
  let point = submit (Sub.make producer [||]) in
  P.interrupt pp;
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      raises Exit (fun () -> Rig.submit run ~waits:[| point |] ~buffers));
  equal ~msg:"both claims released" (pair bool bool) (true, true)
    (donated_exclusive a, donated_exclusive b);
  ignore (P.run pp);
  ignore (Rig.submit run ~waits:[| point |] ~buffers);
  ignore (P.run p)

(* A read whose wait raises leaves no claim: a signal handler's exception
   during the wait. *)
let test_wait_raises () =
  let d, p = P.open_ "claim:wait-raises" in
  let m = B.create d 64 in
  write d m;
  P.interrupt p;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit)) in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () -> raises Exit (fun () -> B.read m Bigarray.char ignore));
  equal ~msg:"no claim left" bool false (claimed m);
  ignore (P.run p)

(* Other memory answers Lost for a point a lost device did not reach, and
   Claimed for one it reached: its stop's last value in the word reaches
   nothing. *)
let test_c_lost_points () =
  let d, p = P.open_ "claim:c-lost-points" in
  let reached = B.create Rig.host (1 lsl 16) in
  let unreached = B.create Rig.host (1 lsl 16) in
  write d (require_some (B.borrow d reached));
  ignore (P.run p);
  write d (require_some (B.borrow d unreached));
  lose d p;
  equal ~msg:"reached" answer R.Claimed (R.claim reached Rig.Read_write);
  R.release reached;
  equal ~msg:"unreached" answer R.Lost (R.claim unreached Rig.Read);
  raises_match (lost_by d) (fun () -> R.wait unreached Rig.Read);
  R.release unreached

(* A lost device's unreached read is followed by a write, not a read. *)
let test_c_lost_reads () =
  let d, p = P.open_ "claim:c-lost-reads" in
  let m = B.create Rig.host (1 lsl 16) in
  read d (require_some (B.borrow d m));
  lose d p;
  equal ~msg:"for reading" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  equal ~msg:"for writing" answer R.Lost (R.claim m Rig.Read_write);
  R.release m

(* A loss behind a live device's unreached work answers Lost, whichever
   device's use came first, and the wait raises the lost device's Lost at
   once: the live device's sleeps block at its gate, so a wait for its work
   would never return. *)
let test_c_lost_behind_live () =
  let order lost_first =
    let name = if lost_first then "claim:c-behind-1" else "claim:c-behind-2" in
    let live, lp = P.open_ (name ^ "-live") in
    let d, p = P.open_ (name ^ "-lost") in
    let m = B.create Rig.host (1 lsl 16) in
    let read_on d = read d (require_some (B.borrow d m)) in
    if lost_first then (read_on d; read_on live) else (read_on live; read_on d);
    lose d p;
    P.gate lp;
    equal ~msg:"for writing" answer R.Lost (R.claim m Rig.Read_write);
    raises_match (lost_by d) (fun () -> R.wait m Rig.Read_write);
    R.release m;
    equal ~msg:"for reading" answer R.Claimed (R.claim m Rig.Read);
    R.release m;
    P.open_gate lp
  in
  order true;
  order false

(* A borrow on a lost device answers Lost; the memory it maps lives on. *)
let test_c_lost_borrow () =
  let d, p = P.open_ "claim:c-lost-borrow" in
  let m = B.create Rig.host (1 lsl 16) in
  let b = require_some (B.borrow d m) in
  lose d p;
  equal ~msg:"the borrow" answer R.Lost (R.claim b Rig.Read);
  raises_match (lost_by d) (fun () -> R.wait b Rig.Read);
  R.release b;
  equal ~msg:"the memory" answer R.Claimed (R.claim m Rig.Read_write);
  R.release m

(* Fixed memory is ordered by its access: a claim to read waits for an
   unreached submission that writes it, and for none that reads it. *)
let test_c_fixed () =
  let d, p = P.open_ "claim:c-fixed" in
  let m = B.create d 64 in
  let fixed access = Rig.Submission.make ~fixed:[ (m, access) ] d [||] in
  ignore (submit (fixed Rig.Read));
  equal ~msg:"an unreached read" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  ignore (submit (fixed Rig.Read_write));
  equal ~msg:"an unreached write" answer R.Wait (R.claim m Rig.Read);
  R.release m;
  ignore (P.run p);
  equal ~msg:"reached" answer R.Claimed (R.claim m Rig.Read);
  R.release m

(* Claims on a lost device's memory raise Lost, and with_ releases what it took
   first. *)
let test_lost_claims () =
  let d, p = P.open_ "claim:lost" in
  let m = B.create d 64 in
  let w = Rig.Submission.make d [| Rig_support.touching [| Rig.Submission.write |] |] in
  ignore (submit w ~buffers:[| m |]);
  P.fail p;
  (try ignore (submit (Rig.Submission.make d [||])) with Rig.Lost _ -> ());
  raises_match lost (fun () -> Claim.with_ ~read:[ m ] ~donate:[] ignore);
  let h = B.create Rig.host 64 in
  raises_match lost (fun () -> Claim.with_ ~read:[ h; m ] ~donate:[] ignore);
  Claim.with_ ~read:[] ~donate:[ h ] (fun c ->
      equal ~msg:"h's claims released" bool true (Claim.exclusive c h))

(* A buffer's death is a fact with its reason: none while it lives, the
   consumption's reason once its memory is consumed; the buffer consume gives
   lives. *)
let test_dead_fact () =
  let b = B.create Rig.host 16 in
  equal ~msg:"live" (option string) None (B.dead b);
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        Claim.consume c ~why:"donated" b)
  in
  equal ~msg:"consumed" (option string) (Some "donated") (B.dead b);
  equal ~msg:"its successor" (option string) None (B.dead b')

(* Host copies beside a donation *)

let text b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

let filled n c =
  let b = B.create Rig.host n in
  B.fill b c;
  b

(* A copy, blit or fill the host runs claims its memory before it waits. In
   another domain, [access] waits at [p]'s gate for work on memory of [p]'s
   device; meanwhile [b] is donated, and a donation that holds its memory
   exclusive gives [donor] the consumed buffer to write in place. It is that
   buffer if the donation held the memory, and how [access] ended. *)
let donate_while_waiting p b access donor =
  P.gate p;
  let ended =
    Domain.spawn (fun () ->
        match access () with
        | () -> "returned"
        | exception e -> Printexc.to_string e)
  in
  Rig_support.await "the host access waiting at the gate" (fun () ->
      P.sleepers p = 1);
  let donated = Rig_support.donated b in
  Option.iter donor donated;
  P.open_gate p;
  (donated, Domain.join ended)

(* Two domains: a wait held at its device's point while a free on the other
   kills its buffer, and the memory returns once the work ran. The wait
   walks the stamps it began on, which live while the dead buffer is
   reachable, so it returns. Run with MallocScribble=1 (macOS) or under
   AddressSanitizer, a walk of freed stamps reads garbage. *)
let test_free_beside_wait () =
  let d, p = P.open_ "claim:free-wait" in
  let h = B.create Rig.host (1 lsl 16) in
  write d (require_some (B.borrow d h));
  P.gate p;
  let waited =
    Domain.spawn (fun () ->
        match B.wait h Rig.Read_write with
        | () -> "returned"
        | exception e -> Printexc.to_string e)
  in
  Rig_support.await "the wait at the gate" (fun () -> P.sleepers p = 1);
  equal ~msg:"freed" bool true (B.free h);
  ignore (P.run p);
  for _ = 1 to 2 do
    ignore (Sys.opaque_identity (B.create Rig.host 0));
    Rig.free_cache d
  done;
  P.open_gate p;
  equal ~msg:"the wait" string "returned" (Domain.join waited);
  ignore (P.run p)

(* Two domains: a borrow held while it maps, and a free on the other. The
   borrow claims the memory while it maps it, so the free changes nothing
   then, and succeeds once the borrow returned. *)
let test_free_beside_borrow () =
  let d, p = P.open_ "claim:free-borrow" in
  let h = B.create Rig.host (1 lsl 16) in
  P.gate_locates p;
  let borrowed =
    Domain.spawn (fun () ->
        match B.borrow d h with
        | Some _ -> "borrowed"
        | None -> "not mapped"
        | exception e -> Printexc.to_string e)
  in
  Rig_support.await "the borrow mapping" (fun () -> P.locaters p = 1);
  equal ~msg:"a free while the borrow maps" bool false (B.free h);
  P.open_locates p;
  equal ~msg:"the borrow" string "borrowed" (Domain.join borrowed);
  equal ~msg:"a free once it returned" bool true (B.free h)

(* The copy reads the source after the wait for its destination's last use. *)
let test_host_copy_source () =
  let d, p = P.open_ "claim:host-copy-source" in
  let src = filled 64 's' and dst = B.create d 64 in
  write d dst;
  let donated, ended =
    donate_while_waiting p src
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  equal ~msg:"the copy" string "returned" ended;
  equal ~msg:"donated during the copy" bool false (Option.is_some donated);
  equal ~msg:"the copy's bytes" string (String.make 64 's') (text dst)

(* The copy writes the destination after the wait for its source's last
   write. *)
let test_host_copy_destination () =
  let d, p = P.open_ "claim:host-copy-destination" in
  let src = B.create d 64 in
  B.fill src 's';
  write d src;
  let dst = filled 64 'd' in
  let donated, ended =
    donate_while_waiting p dst
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  Option.iter
    (fun b -> equal ~msg:"the donor's bytes" string (String.make 64 'X') (text b))
    donated;
  equal ~msg:"the copy" string "returned" ended;
  equal ~msg:"donated during the copy" bool false (Option.is_some donated);
  equal ~msg:"the copy's bytes" string (String.make 64 's') (text dst)

(* A blit or fill on memory of [d] the host addresses waits for the work on
   it, and keeps a donation to a read claim meanwhile. *)
let host_accesses =
  [
    ("a blit to bytes", fun b -> B.blit_to_bytes b 0 (Bytes.create 64) 0 64);
    ("a blit from a string", fun b -> B.blit_from_string (String.make 64 'a') 0 b 0 64);
    ("a fill", fun b -> B.fill b 'f');
  ]

let test_host_access_claims () =
  List.iter
    (fun (name, access) ->
      let d, p = P.open_ ("claim:host-access-" ^ name) in
      let b = B.create d 64 in
      write d b;
      let donated, ended =
        donate_while_waiting p b (fun () -> access b) ignore
      in
      equal ~msg:(name ^ " returned") string "returned" ended;
      equal ~msg:(name ^ ", donated during it") bool false
        (Option.is_some donated))
    host_accesses

(* A donor submits work on the buffer it consumed before its with_ returns:
   the work writes the memory in place, and the host reads the write. *)
let test_donor_submits () =
  let module Sub = Rig.Submission in
  let d, _ = P.open_ "claim:donor-submits" in
  let b = B.create d 8 in
  Rig_support.store (R.host b) 0;
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        let b' = Claim.consume c ~why:"donated" b in
        let s =
          Sub.make ~fixed:[ (b', Rig.Read_write) ] d
            [| Sub.fill Rig_support.poke ~params:16 |]
        in
        let run = Sub.Run.make s in
        Sub.Run.int64 run (Sub.block s 0) 0 (Int64.of_int (Sub.address s b'));
        Sub.Run.int64 run (Sub.block s 0) 8 42L;
        Rig.Point.wait (Rig.submit run ~buffers:[||]);
        b')
  in
  equal ~msg:"the word the work stored" int 42
    (B.read b' Bigarray.int64 (fun a -> Int64.to_int a.{0}))

(* The consumed buffer *)

type place = Host | Addressed | Unaddressed

let pp_place ppf = function
  | Host -> Format.pp_print_string ppf "host"
  | Addressed -> Format.pp_print_string ppf "addressed"
  | Unaddressed -> Format.pp_print_string ppf "unaddressed"

let addressed = lazy (fst (P.open_ "claim:consumed-addressed"))

let unaddressed =
  lazy (fst (P.open_ ~host_visible:false "claim:consumed-unaddressed"))

let device = function
  | Host -> Rig.host
  | Addressed -> Lazy.force addressed
  | Unaddressed -> Lazy.force unaddressed

let consumed_cases =
  List.concat_map
    (fun place -> List.map (fun n -> (place, n)) [ 0; 1; 8; 4096 ])
    [ Host; Addressed; Unaddressed ]

let consumed_name (place, n) =
  Format.asprintf "%a memory of %d bytes" pp_place place n

(* [exclusive c b] holds until [c] consumes [b], and is false after it for
   both buffers. The consumed buffer is then used as any buffer its holder
   made: work submitted on it, the host's reads and writes, a copy a device
   runs, a share. *)
let consumed_law (place, n) =
  let d = device place in
  let b = B.create d n in
  let other = B.create (device Unaddressed) n in
  let b' =
    Claim.with_ ~read:[] ~donate:[ b ] @@ fun c ->
    equal ~msg:"before the consumption" bool true (Claim.exclusive c b);
    let b' = Claim.consume c ~why:"consumed" b in
    equal ~msg:"after, the consumed buffer" (pair bool bool) (false, false)
      (Claim.exclusive c b, Claim.exclusive c b');
    if place <> Host then write d b';
    (* A device's memory of no bytes has no host address. *)
    if place = Host || (place = Addressed && n > 0) then begin
      B.write b' Bigarray.char (fun a -> Bigarray.Array1.fill a 'w');
      equal ~msg:"read back" string (String.make n 'w')
        (B.read b' Bigarray.char (fun a ->
             String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a)))
    end
    else B.fill b' 'w';
    B.copy ~src:b' ~dst:other;
    Claim.share b';
    b'
  in
  equal ~msg:"the copy" string (String.make n 'w') (text other);
  equal ~msg:"shared for good" bool false (donated_exclusive b')

(* Two domains: host copies, blits and fills of a memory beside donations of
   it. A donation that holds the memory exclusive consumes it, fills the
   buffer it gets and reads it back. Every access writes or reads the byte the
   memory was made with, so while the memory is live they agree whatever their
   order. An access after a consumption finds the buffer dead; one beside a
   donation may find it held exclusive, which an order with the donation first
   explains. A donation may also find the memory read by an access beside it
   and answer [None], in any state, which no order of whole calls shows: an
   access's claim spans its call. *)
type copied = { mutable gone : bool }

let copied =
  abstract ~pp:(fun ppf r -> Format.fprintf ppf "dead %b" r.gone) "m"

let made = 's'

let judge_bytes r = function
  | Ok s ->
      equal ~msg:"dead" bool false r.gone;
      equal ~msg:"bytes" string (String.make 64 made) s
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.gone
  | Error e -> raise e

let judge_access r = function
  | Ok () -> equal ~msg:"dead" bool false r.gone
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.gone
  | Error e -> raise e

let judge_donor r = function
  | Ok None -> equal ~msg:"dead" bool false r.gone
  | Ok (Some s) ->
      equal ~msg:"dead" bool false r.gone;
      equal ~msg:"the donor's bytes" string (String.make 64 'X') s;
      r.gone <- true
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.gone
  | Error e -> raise e

let copy_commands =
  [
    command "make"
      (Gen.unit @-> makes copied)
      (fun () -> { gone = false })
      (fun () -> filled 64 made);
    command "copy out" (copied ^-> judges string) judge_bytes (fun b ->
        let dst = B.create Rig.host 64 in
        B.copy ~src:b ~dst;
        text dst);
    command "copy in" (copied ^-> judges unit) judge_access (fun b ->
        B.copy ~src:(filled 64 made) ~dst:b);
    command "blit out" (copied ^-> judges string) judge_bytes text;
    command "blit in" (copied ^-> judges unit) judge_access (fun b ->
        B.blit_from_string (String.make 64 made) 0 b 0 64);
    command "fill" (copied ^-> judges unit) judge_access (fun b ->
        B.fill b made);
    command "donate"
      (copied ^-> judges (option string))
      judge_donor
      (fun b ->
        Rig_support.donated b
        |> Option.map (fun b' ->
               B.fill b' 'X';
               text b'));
  ]

let tests =
  [
    group ~timeout "claims"
      [
        prop
          ~examples:
            [
              (* A donation overlapping a read that begins before a shorter
                 read: the overlap is not with the read next to it in memory
                 order. *)
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
        test "donations that overlap each other are refused"
          test_donations_overlap;
        test "memories with no address are claimed apart" test_io_spans;
        test "readers share a memory, an exclusive claim excludes them"
          test_claims;
        test "a donor uses memory it holds exclusive once it consumes it"
          test_exclusive_writes;
        test "a borrow and the memory it maps share one count"
          test_borrow_counts;
        test "a borrow of memory held exclusive raises" test_borrow_exclusive;
        test "a bigarray's memory is never exclusive" test_of_bigarray;
        test "memory shared through any of its buffers is never exclusive"
          test_share_covers;
        test "each buffer of a donation is exclusive on its own"
          test_shards;
        test "an export under an exclusive claim waits for the consumption"
          test_export_exclusive;
        test "a donor submits work on the consumed buffer inside with_"
          test_donor_submits;
        cases ~name:consumed_name
          "exclusive holds until the consumption, and the consumed buffer is \
           used as any other"
          consumed_cases consumed_law;
      ];
    group ~timeout "consumption"
      [
        test "memory consumed twice lives while its last buffer does"
          test_consume_twice;
        test "a buffer consumed while another claim reads admits only reads"
          test_consume_read;
        test "only a claimed buffer that spans its memory is consumed"
          test_consume_refusals;
        test "a dead buffer refuses claims and accepts a release"
          test_dead_claims;
        test "the C claim of a closed device's buffer answers its loss"
          test_closed_slot_claim;
        test "sharing a dead buffer or memory held exclusive is refused"
          test_share_refused;
        test "a claim whose with_ returned holds nothing" test_stale_returned;
        test "a claim whose with_ raised holds nothing" test_stale_raised;
        test "a signal pending as with_'s function returns leaves no claim"
          test_signal_returned;
        test "a signal pending as with_'s function raises leaves no claim"
          test_signal_raised;
        test "a buffer's death is a fact with its reason" test_dead_fact;
        test "claims on memory a loss reaches raise and release"
          test_lost_claims;
      ];
    group ~timeout "holds"
      [
        test "a hold keeps memory from donation until it is collected"
          test_hold;
        test "a hold of a dead buffer or memory held exclusive holds nothing"
          test_hold_refusals;
        test "a release ends a hold before its collection" test_release;
        test "a later release and the collection of a released hold end \
              nothing"
          test_release_once;
        stateful ~domains:2
          "releases of one hold on two domains beside its collection end it \
           once"
          release_commands;
        test "a submission holds the memory every run reads" test_submission_holds;
        test "consumable says whether a consumption succeeds" test_consumable;
        stateful ~domains:2
          "holds beside consumptions of one memory are ordered with them"
          hold_commands;
      ];
    group ~timeout "claims from C"
      [
        prop "rig_buffers_claim claims every buffer or none" states
          claim_all_law;
        test "a claim from C is a read claim" test_c_claims;
        test
          "a claim from C holds the memory and waits until the word shows \
           its points"
          test_c_wait;
        test "behind a transport a claim reads the word a wait read"
          test_c_transport;
        test "a claim on a lost device's memory answers Lost under the claim"
          test_c_lost;
        test "a device lost during a wait from C raises Lost under the claim"
          test_c_lost_waiting;
        test "wait claims, then waits for the work the access follows"
          test_wait;
        test "a read whose wait raises releases its claim" test_wait_raises;
        test "a claim on memory a lost device did not reach answers Lost"
          test_c_lost_points;
        test "a lost device's unreached read is lost for writing only"
          test_c_lost_reads;
        test "a loss behind a live device's unreached work answers Lost"
          test_c_lost_behind_live;
        test "a borrow on a lost device answers Lost" test_c_lost_borrow;
        test "a claim on fixed memory follows its uses by their access"
          test_c_fixed;
      ];
    group ~timeout "domains"
      [
        test "fixed memory prevents donation during submit"
          (test_submit_donation ("fixed", `Fixed));
        test "a run's buffers prevent donation during submit"
          (test_submit_donation ("copy", `Copy));
        test "a run's refused buffer releases every claim the submit took"
          test_run_claim_refusals;
        test "an exceptional submit wait releases its run's claims"
          test_run_wait_raises;
        stateful ~domains:2
          "with_ of a group and reads of its memory leave no claim behind"
          group_commands;
        stateful ~domains:2
          "reads, exports and shares beside a consuming donation are ordered \
           with it"
          donation_commands;
        test "a wait beside a free walks the stamps it began on"
          test_free_beside_wait;
        test "a free beside a borrow that maps changes nothing"
          test_free_beside_borrow;
        test "a host copy waiting holds its source from a donation"
          test_host_copy_source;
        test "a host copy waiting holds its destination from a donation"
          test_host_copy_destination;
        test "a host blit or fill waiting holds its memory from a donation"
          test_host_access_claims;
        stateful ~domains:2
          "host copies, blits and fills beside donations are ordered with \
           them"
          copy_commands;
        stateful ~domains:2
          "a consumption through handed claims beside their return and \
           donations is ordered with them"
          handed_commands;
      ];
  ]

let () = exit (run "rig.claim" tests)
