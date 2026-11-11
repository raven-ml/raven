(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
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
  Rig.submit ?waits (Rig.Submission.take s) ~buffers

let write d m =
  let s =
    Rig.Submission.make d [| Rig_support.touching [| Rig.Submission.write |] |]
  in
  ignore (submit s ~buffers:[| m |])

let read d m =
  let s =
    Rig.Submission.make d [| Rig_support.touching [| Rig.Submission.read |] |]
  in
  ignore (submit s ~buffers:[| m |])

let lost = function Rig.Lost _ -> true | _ -> false
let lost_by d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* Loses [d]: its next submit fails. *)
let lose d p =
  P.fail p;
  try ignore (submit (Rig.Submission.make d [||])) with Rig.Lost _ -> ()

let timeout = 60.

(* The cases of each law that covers labels. The rarest label marks about one
   case in sixteen: 300 cases miss it with odds under one in a hundred
   million, where 100 missed it in about one run in 600. *)
let law_cases = 300
let answer = Testable.make ~pp:R.pp_answer ~equal:( = )

let access =
  Testable.make
    ~pp:(fun ppf a ->
      Format.pp_print_string ppf
        (match a with Rig.Read -> "Read" | Rig.Read_write -> "Read_write"))
    ~equal:( = )

let dead why = Exn.invalid_arg ~sub:why
let live bs = List.for_all (fun b -> Option.is_none (B.dead b)) bs

let text b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

let filled n c =
  let b = B.create Rig.host n in
  B.fill b c;
  b

(* The buffer a take of [b] alone gives, with the reason [why]. *)
let taken ?(why = "donated") b =
  match B.take ~why [ b ] with
  | Some [ b' ] -> b'
  | Some bs -> failf "a take of one buffer gave %d" (List.length bs)
  | None -> fail "a take of memory nothing claims was refused"

(* Whether a take of [b] alone succeeds, which kills [b]. *)
let takes b = Option.is_some (Rig_support.donated b)

(* Whether a claim holds [m]'s memory: a take of it alone is [None], which
   changes nothing. Where none does, the take kills [m]. *)
let claimed m = not (takes m)

(* Takes of views of several memories, against a model *)

(* What uses a memory before the take: nothing, a read claim, a keep, an
   export, or a share through a view of its last byte. *)
type use = Unused | Read_claim | Kept | Exported | Shared

(* A case: the bytes of each memory, each memory's use, and the take's
   buffers, each a view of a memory: its index, first byte and length. *)
type case = { n : int; uses : use list; views : (int * int * int) list }

let pp_use ppf u =
  Format.pp_print_string ppf
    (match u with
    | Unused -> "unused"
    | Read_claim -> "read"
    | Kept -> "kept"
    | Exported -> "exported"
    | Shared -> "shared")

let pp_case ppf c =
  Format.fprintf ppf "%d bytes, uses [%a], views %s" c.n
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_use)
    c.uses
    (String.concat " "
       (List.map
          (fun (m, o, l) -> Printf.sprintf "[%d: %d,+%d]" m o l)
          c.views))

let case =
  let open Gen in
  let unused k = List.init k (fun _ -> Unused) in
  let wholes n k = List.init k (fun m -> (m, 0, n)) in
  (* Memories nothing uses, each named once by a view of all of it. *)
  let clean n =
    map (fun k -> { n; uses = unused k; views = wholes n k }) (int_range 0 3)
  in
  (* One memory of several used. *)
  let used n =
    bind (int_range 1 3) (fun k ->
        map
          (fun (i, u) ->
            {
              n;
              uses = List.init k (fun j -> if j = i then u else Unused);
              views = wholes n k;
            })
          (pair (int_range 0 (k - 1))
             (of_list [ Read_claim; Kept; Exported; Shared ])))
  in
  (* The first memory named twice. *)
  let twice n =
    map
      (fun k -> { n; uses = unused k; views = (0, 0, n) :: wholes n k })
      (int_range 1 3)
  in
  (* The first memory named by a view of part of it, which a memory of no
     bytes has none of. *)
  let part n =
    if n = 0 then clean n
    else
      bind (pair (int_range 1 3) (int_range 0 n)) (fun (k, o) ->
          map
            (fun l ->
              let l = if o = 0 && l = n then n - 1 else l in
              { n; uses = unused k; views = (0, o, l) :: List.tl (wholes n k) })
            (int_range 0 (n - o)))
  in
  let use =
    frequency
      [
        (4, constant Unused);
        (1, constant Read_claim);
        (1, constant Kept);
        (1, constant Exported);
        (1, constant Shared);
      ]
  in
  let view n k =
    bind (int_range 0 (k - 1)) (fun m ->
        frequency
          [
            (3, constant (m, 0, n));
            ( 1,
              bind (int_range 0 n) (fun o ->
                  map (fun l -> (m, o, l)) (int_range 0 (n - o))) );
          ])
  in
  let any n =
    bind (list ~size:(int_range 1 4) use) (fun uses ->
        map
          (fun views -> { n; uses; views })
          (list ~size:(int_range 0 4) (view n (List.length uses))))
  in
  with_pp pp_case
    (bind
       (frequency [ (1, constant 0); (1, constant 1); (2, int_range 2 32) ])
       (fun n ->
         frequency
           [ (2, clean n); (2, used n); (1, twice n); (1, part n); (2, any n) ]))

let spans_all c = List.for_all (fun (_, o, l) -> o = 0 && l = c.n) c.views

let named_once c =
  let named = List.map (fun (m, _, _) -> m) c.views in
  List.length (List.sort_uniq Int.compare named) = List.length named

let named_use c u = List.exists (fun (m, _, _) -> List.nth c.uses m = u) c.views

(* What a take must do: take every memory when each view spans its memory,
   no memory is named twice, and nothing uses a memory named. *)
let takes_all c =
  spans_all c && named_once c
  && List.for_all (fun (m, _, _) -> List.nth c.uses m = Unused) c.views

let take_law c =
  let made i = Char.chr (Char.code 'a' + i) in
  let memories = List.mapi (fun i _ -> filled c.n (made i)) c.uses in
  (* Starts [u] on [b]'s memory, and is how it ends: an export or a share
     lasts for good. *)
  let start u b =
    match u with
    | Unused -> ignore
    | Read_claim ->
        claim_read b;
        fun () -> claim_release b
    | Kept ->
        let k = B.keep [ b ] in
        fun () -> B.release k
    | Exported ->
        ignore (B.export b Bigarray.char);
        ignore
    | Shared ->
        B.share (B.view b ~first:(max 0 (c.n - 1)) ~length:(min c.n 1));
        ignore
  in
  let ends = List.map2 start c.uses memories in
  let buffers =
    List.map
      (fun (m, first, length) -> B.view (List.nth memories m) ~first ~length)
      c.views
  in
  let named m = List.exists (fun (m', _, _) -> m' = m) c.views in
  let e = takes_all c in
  let whole = spans_all c && named_once c in
  let unused =
    List.for_all (fun (m, _, _) -> List.nth c.uses m = Unused) c.views
  in
  cover "several memories taken" (e && List.length c.views >= 2);
  cover "no buffer taken" (c.views = []);
  cover "a memory of no bytes taken" (e && c.n = 0 && c.views <> []);
  cover "refused for a view of part of a memory"
    ((not (spans_all c)) && named_once c && unused);
  cover "refused for a memory named twice"
    (spans_all c && (not (named_once c)) && unused);
  cover "refused by a read claim" (whole && named_use c Read_claim);
  cover "refused by a keep" (whole && named_use c Kept);
  cover "refused outside the claims"
    (whole && (named_use c Exported || named_use c Shared));
  let got = B.take ~why:"taken" buffers in
  equal ~msg:"taken" bool e (Option.is_some got);
  (match got with
  | None ->
      List.iteri
        (fun i b ->
          let msg = Printf.sprintf "memory %d" i in
          equal ~msg (option string) None (B.dead b);
          equal ~msg string (String.make c.n (made i)) (text b))
        memories
  | Some bs ->
      List.iteri
        (fun i b ->
          equal
            ~msg:(Printf.sprintf "memory %d's buffer" i)
            (option string)
            (if named i then Some "taken" else None)
            (B.dead b))
        memories;
      List.iter
        (fun v ->
          equal ~msg:"a view taken" (option string) (Some "taken") (B.dead v))
        buffers;
      equal ~msg:"the buffers given" int (List.length buffers)
        (List.length bs);
      List.iter2
        (fun (m, _, _) b ->
          let msg = Printf.sprintf "memory %d's new buffer" m in
          equal ~msg string (String.make c.n (made m)) (text b);
          equal ~msg bool true (B.spans b);
          equal ~msg access Rig.Read_write (B.access b))
        c.views bs);
  List.iter (fun f -> f ()) ends;
  (* A refused take left no claim: each memory still live, its use ended,
     takes alone unless it is outside the claims. *)
  List.iteri
    (fun i b ->
      if B.dead b = None then
        equal
          ~msg:(Printf.sprintf "memory %d taken alone after" i)
          bool
          (match List.nth c.uses i with
          | Exported | Shared -> false
          | Unused | Read_claim | Kept -> true)
          (takes b))
    memories

(* Takes *)

(* A take of [c] and [b] that a use of [b]'s memory refuses: [None], with
   both live and their bytes as they were. [use b] starts the use and is how
   it ends, if it does: the take then succeeds, so the refused one left no
   claim. A use that lasts leaves [c] alone to take. *)
let refused name make use =
  let c = filled 8 'c' and b = make () in
  let ended = use b in
  is_none ~msg:name (B.take ~why:"taken" [ c; b ]);
  equal ~msg:(name ^ ": live") bool true (live [ c; b ]);
  equal ~msg:(name ^ ": c's bytes") string "cccccccc" (text c);
  match ended with
  | Some ended ->
      ended ();
      is_some ~msg:(name ^ ", once it ended") (B.take ~why:"taken" [ c; b ])
  | None -> is_some ~msg:(name ^ ", c alone") (B.take ~why:"taken" [ c ])

let host () = filled 8 'b'

let test_take_refused () =
  refused "a keep" host (fun b ->
      let k = B.keep [ b ] in
      Some (fun () -> B.release k));
  refused "a share" host (fun b ->
      B.share b;
      None);
  refused "an export" host (fun b ->
      ignore (B.export b Bigarray.char);
      None);
  refused "a read claim in progress" host (fun b ->
      claim_read b;
      Some (fun () -> claim_release b));
  let io = Rig_support.io "claim:take-read-only" in
  refused "memory that admits only reads"
    (fun () -> Rig_support.io_buffer ~access:Rig.Read io 8)
    (fun _ -> None)

(* A take inside a read of one of its buffers, whose claim lasts the call, is
   refused, and succeeds once the read returned. *)
let test_take_reading () =
  let c = filled 8 'c' and b = filled 8 'b' in
  B.read b Bigarray.char (fun _ ->
      is_none ~msg:"inside the read" (B.take ~why:"taken" [ c; b ]));
  equal ~msg:"live" bool true (live [ c; b ]);
  is_some ~msg:"once it returned" (B.take ~why:"taken" [ c; b ])

(* Two buffers of a take over one memory each claim the other's: [None], and
   every buffer stays live, whether the second is the same buffer, a view of
   all of the memory or a borrow of it. A view of part of the memory does not
   span it. *)
let test_take_overlap () =
  let d = Result.get_ok (Rig.software "claim:take-overlap") in
  let n = 1 lsl 16 in
  let c = filled 8 'c' and b = B.create Rig.host n in
  List.iter
    (fun (msg, b') ->
      is_none ~msg (B.take ~why:"taken" [ c; b; b' ]);
      equal ~msg:(msg ^ ": live") bool true (live [ c; b; b' ]))
    [
      ("the buffer twice", b);
      ("a view of all of it", B.view b ~first:0 ~length:n);
      ("a borrow of it", require_some (B.borrow d b));
    ];
  is_none ~msg:"a view of part of it"
    (B.take ~why:"taken" [ c; B.view b ~first:0 ~length:8 ]);
  is_some ~msg:"each once" (B.take ~why:"taken" [ c; b ])

(* A take of several memories takes each: every earlier buffer over them is
   dead with its reason, a view included, and the take gives one live buffer
   per memory, in order, which admits writes. A take of none succeeds. *)
let test_take () =
  let a = filled 8 'a' and b = filled 8 'b' in
  let view = B.view b ~first:2 ~length:4 in
  (match B.take ~why:"taken" [ a; b ] with
  | Some [ a'; b' ] ->
      equal ~msg:"the earlier buffers" (list (option string))
        [ Some "taken"; Some "taken"; Some "taken" ]
        [ B.dead a; B.dead b; B.dead view ];
      equal ~msg:"in order" (pair string string) ("aaaaaaaa", "bbbbbbbb")
        (text a', text b');
      B.fill a' 'x';
      B.fill b' 'y';
      equal ~msg:"written in place" (pair string string)
        ("xxxxxxxx", "yyyyyyyy") (text a', text b')
  | Some bs -> failf "%d buffers taken of 2" (List.length bs)
  | None -> fail "two memories nothing claims refused");
  equal ~msg:"none" (option int) (Some 0)
    (Option.map List.length (B.take ~why:"taken" []))

(* Memories of one device that has no address of them are told apart all the
   same: two are taken together, one twice is refused. *)
let test_no_address () =
  let d, _ =
    P.open_ ~addresses:false ~host_visible:false "claim:no-address"
  in
  equal ~msg:"the host addresses none" bool false (Rig.reaches Rig.host d);
  let a = B.create d 64 and b = B.create d 64 in
  is_none ~msg:"one twice" (B.take ~why:"taken" [ a; B.view a ~first:0 ~length:64 ]);
  is_some ~msg:"both" (B.take ~why:"taken" [ a; b ])

(* A taker uses the memory it took through the buffer the take gives. The
   host's reads and writes through an earlier buffer and every copy, blit or
   fill of one, whether the host, a device or staging memory runs it, raise
   with the take's reason, and each of them runs on the new buffer. *)
let test_taken_writes () =
  let d, _ = P.open_ ~host_visible:false "claim:taken-writes" in
  let staging, _ =
    P.open_ ~maps_host:false ~host_visible:false "claim:taken-staged"
  in
  let src = B.of_string "abcdefgh" in
  let refused msg f = raises_match ~msg (dead "donated") f in
  let earlier b =
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
  let used b =
    let b' = taken b in
    earlier b;
    B.copy ~src ~dst:b';
    B.blit_from_string "ijkl" 0 b' 4 4;
    B.fill (B.view b' ~first:0 ~length:2) 'z';
    let s = B.create staging 8 in
    B.copy ~src:b' ~dst:s;
    B.copy ~src:s ~dst:b';
    text b'
  in
  equal string ~msg:"host memory" "zzcdijkl" (used (B.create Rig.host 8));
  equal string ~msg:"memory the host does not address" "zzcdijkl"
    (used (B.create d 8))

(* A borrow and the memory it maps share one count: a claim through the
   borrow refuses a take of the memory until it ends. *)
let test_borrow_counts () =
  let d = Result.get_ok (Rig.software "claim:borrow") in
  let b = B.create Rig.host (1 lsl 16) in
  let borrowed = require_some (B.borrow d b) in
  claim_read borrowed;
  is_none ~msg:"under the claim" (B.take ~why:"taken" [ b ]);
  claim_release borrowed;
  is_some ~msg:"once it ended" (B.take ~why:"taken" [ b ])

(* A borrow of a buffer a take killed raises with the take's reason, and one
   of the buffer it gave maps the memory. *)
let test_borrow_taken () =
  let d = Result.get_ok (Rig.software "claim:borrow-taken") in
  let b = B.create Rig.host (1 lsl 16) in
  let b' = taken b in
  raises_match (dead "donated") (fun () -> B.borrow d b);
  is_some ~msg:"the taken buffer" (B.borrow d b')

let test_of_bigarray () =
  let b =
    B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8)
  in
  is_none (B.take ~why:"taken" [ b ]);
  equal ~msg:"live" (option string) None (B.dead b)

(* Memory shared through one buffer is never taken through another: a view,
   a borrow, the buffer itself. Sharing it again changes nothing, and readers
   claim and release it as before. *)
let test_share_covers () =
  let d = Result.get_ok (Rig.software "claim:share") in
  let fresh () =
    let b = B.create Rig.host (1 lsl 16) in
    (b, B.view b ~first:0 ~length:8, require_some (B.borrow d b))
  in
  let b, view, borrowed = fresh () in
  B.share view;
  equal ~msg:"shared through a view, the buffer" bool false (takes b);
  equal ~msg:"shared through a view, the borrow" bool false (takes borrowed);
  B.share view;
  B.share b;
  equal ~msg:"shared again" bool false (takes b);
  claim_read b;
  claim_release b;
  let b, _, borrowed = fresh () in
  B.share borrowed;
  equal ~msg:"shared through the borrow" bool false (takes b);
  let b, _, borrowed = fresh () in
  B.share b;
  equal ~msg:"shared through the buffer" bool false (takes borrowed);
  let b, _, _ = fresh () in
  equal ~msg:"unshared" bool true (takes b)

(* A take of memories of two devices takes both, or neither while one is
   claimed. *)
let test_shards () =
  let d = Result.get_ok (Rig.software "claim:shards") in
  let a = B.create Rig.host 8 and b = B.create d 8 in
  claim_read b;
  is_none ~msg:"one claimed" (B.take ~why:"taken" [ a; b ]);
  equal ~msg:"both live" bool true (live [ a; b ]);
  claim_release b;
  is_some ~msg:"neither claimed" (B.take ~why:"taken" [ a; b ])

(* An export of taken memory is the taker's: refused through the earlier
   buffer, made through the new one, and the memory is outside the claims
   after it. *)
let test_export_taken () =
  let b = B.create Rig.host 16 in
  let b' = taken b in
  raises_match ~msg:"the earlier buffer" (dead "donated") (fun () ->
      B.export b Bigarray.char);
  Bigarray.Array1.fill (B.export b' Bigarray.char) 'x';
  equal ~msg:"its bytes" string (String.make 16 'x') (text b');
  equal ~msg:"outside the claims" bool false (takes b')

(* A taker submits work on the buffer the take gave: the work writes the
   memory in place, and the host reads the write. *)
let test_taker_submits () =
  let module Sub = Rig.Submission in
  let d, _ = P.open_ "claim:taker-submits" in
  let b = B.create d 8 in
  Rig_support.store (R.host b) 0;
  let b' = taken b in
  let s =
    Sub.make ~fixed:[ (b', Rig.Read_write) ] d
      [| Sub.fill Rig_support.poke ~params:16 |]
  in
  let run = Sub.take s in
  Sub.Run.int64 run (Sub.block s 0) 0 (Int64.of_int (Sub.address s b'));
  Sub.Run.int64 run (Sub.block s 0) 8 42L;
  Rig.Point.wait (Rig.submit run ~buffers:[||]);
  equal ~msg:"the word the work stored" int 42
    (B.read b' Bigarray.int64 (fun a -> Int64.to_int a.{0}))

(* The buffer a take gives *)

type place = Host | Addressed | Unaddressed

let pp_place ppf = function
  | Host -> Format.pp_print_string ppf "host"
  | Addressed -> Format.pp_print_string ppf "addressed"
  | Unaddressed -> Format.pp_print_string ppf "unaddressed"

let addressed = lazy (fst (P.open_ "claim:taken-addressed"))

let unaddressed =
  lazy (fst (P.open_ ~host_visible:false "claim:taken-unaddressed"))

let device = function
  | Host -> Rig.host
  | Addressed -> Lazy.force addressed
  | Unaddressed -> Lazy.force unaddressed

let taken_cases =
  List.concat_map
    (fun place -> List.map (fun n -> (place, n)) [ 0; 1; 8; 4096 ])
    [ Host; Addressed; Unaddressed ]

let taken_name (place, n) =
  Format.asprintf "%a memory of %d bytes" pp_place place n

(* The buffer a take gives is used as any buffer its holder made: work
   submitted on it, the host's reads and writes, a copy a device runs, a
   share. *)
let taken_law (place, n) =
  let d = device place in
  let b = B.create d n in
  let other = B.create (device Unaddressed) n in
  let b' = taken ~why:"taken" b in
  equal ~msg:"the earlier buffer" (option string) (Some "taken") (B.dead b);
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
  B.share b';
  equal ~msg:"the copy" string (String.make n 'w') (text other);
  equal ~msg:"shared for good" bool false (takes b')

(* Taken memory *)

(* Memory taken twice lives while the last buffer over it is reachable, and
   every earlier buffer is dead. *)
let test_take_twice () =
  let b = B.create Rig.host (1 lsl 16) in
  let view = B.view b ~first:0 ~length:8 in
  let b' = taken ~why:"first" b in
  let b'' = taken ~why:"second" b' in
  raises_match Exn.invalid_arg (fun () -> B.wait b Rig.Read);
  raises_match Exn.invalid_arg (fun () -> B.wait view Rig.Read);
  raises_match (dead "second") (fun () -> B.wait b' Rig.Read);
  B.write b'' Bigarray.char (fun a -> Bigarray.Array1.fill a 'c');
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create Rig.host (1 lsl 16)));
  equal char 'c' (B.read b'' Bigarray.char (fun a -> a.{(1 lsl 16) - 1}))

(* A taken memory's dead buffers refuse claims from C and raise for a read
   and a keep. The buffer the take gave is claimed. *)
let test_dead_claims () =
  let b = B.create Rig.host 8 in
  let live = taken b in
  equal ~msg:"a claim from C" answer R.Dead (R.claim b Rig.Read);
  raises_match ~msg:"a read" (dead "donated") (fun () ->
      B.read b Bigarray.char ignore);
  raises_match ~msg:"a keep" (dead "donated") (fun () -> B.keep [ b ]);
  claim_read live;
  claim_release live

(* A buffer of a closed device answers that device's loss, also once a later
   device holds its slot: the C claim reads the slot as the device's only while
   the slot's line names it. The later device's first value shows that it took
   the slot. *)
let test_closed_slot_claim () =
  let d, _ = P.open_ "claim:closed" in
  let p =
    Rig.submit (Rig.Submission.take (Rig.Submission.make d [||]))
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

(* Sharing a dead buffer raises and changes nothing. The buffer a take gives
   is shared as any other. *)
let test_share_refused () =
  let b = B.create Rig.host 16 in
  let b' = taken b in
  raises_match ~msg:"dead" (dead "donated") (fun () -> B.share b);
  let b'' = taken ~why:"again" b' in
  B.share b'';
  raises_match ~msg:"dead, its memory shared" (dead "again") (fun () ->
      B.share b');
  equal ~msg:"the live buffer, shared" bool false (takes b'')

(* A buffer's death is a fact with its reason: none while it lives, the take's
   reason once its memory is taken; the buffer the take gives lives. *)
let test_dead_fact () =
  let b = B.create Rig.host 16 in
  equal ~msg:"live" (option string) None (B.dead b);
  let b' = taken b in
  equal ~msg:"taken" (option string) (Some "donated") (B.dead b);
  equal ~msg:"its successor" (option string) None (B.dead b')

(* A take of a lost device's memory raises Lost wherever the buffer lies in
   the list, and takes nothing. A claim on another buffer of the take makes
   it [None] instead, in either order: it would take nothing. *)
let test_lost_take () =
  let d, p = P.open_ "claim:lost" in
  let m = B.create d 64 in
  write d m;
  lose d p;
  let h = B.create Rig.host 64 in
  raises_match ~msg:"alone" lost (fun () -> B.take ~why:"taken" [ m ]);
  raises_match ~msg:"after a live buffer" lost (fun () ->
      B.take ~why:"taken" [ h; m ]);
  raises_match ~msg:"before a live buffer" lost (fun () ->
      B.take ~why:"taken" [ m; h ]);
  claim_read h;
  is_none ~msg:"before a claimed buffer" (B.take ~why:"taken" [ m; h ]);
  is_none ~msg:"after a claimed buffer" (B.take ~why:"taken" [ h; m ]);
  claim_release h;
  equal ~msg:"the live buffer" bool true (takes h)

(* A signal pending as a read's function returns or raises lands after the
   read ended its claim. [f]'s last act raises SIGINT, which the runtime
   records; its handler raises [Exit] at the next poll point, which the
   allocation after the call places if the read has none left. *)
exception Raised

let interrupted_at_end f =
  let b = B.create Rig.host 8 in
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      raises Exit (fun () ->
          (match B.read b Bigarray.char f with
          | () -> ()
          | exception Raised -> ());
          ignore (Sys.opaque_identity (ref ()))));
  equal ~msg:"taken" bool true (takes b)

let test_signal_returned () =
  interrupted_at_end (fun _ -> Rig_support.interrupt ())

let test_signal_raised () =
  interrupted_at_end (fun _ ->
      Rig_support.interrupt ();
      raise_notrace Raised)

(* A signal pending as the function of a write of Mapped memory returns or
   raises lands once the write's flush ran and its count and claim ended. *)
let interrupted_window name f =
  let d, p = P.open_ ~host_visible:false ("claim:signal-" ^ name) in
  let m = B.create ~memory:Mapped d 8 in
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      raises Exit (fun () ->
          (match B.write m Bigarray.char f with
          | () -> ()
          | exception Raised -> ());
          ignore (Sys.opaque_identity (ref ()))));
  equal ~msg:"counted calls" int 0 (Rig_support.counted ());
  equal ~msg:"taken" bool true (takes m);
  equal ~msg:"flushes" (list string) [ "flush" ]
    (List.filter (String.equal "flush") (P.log p))

let test_signal_window_returned () =
  interrupted_window "window-returned" (fun a ->
      a.{0} <- 'w';
      Rig_support.interrupt ())

let test_signal_window_raised () =
  interrupted_window "window-raised" (fun a ->
      a.{0} <- 'w';
      Rig_support.interrupt ();
      raise_notrace Raised)

(* An exception at any allocation of a blit or a fill into Mapped memory
   outside its counted call, as an asynchronous one lands, leaves no claim
   and no count. Each landing's check takes the memory, and the next landing
   uses the buffer the take gave. *)
let walk_window name call =
  let d, _ = P.open_ ~host_visible:false ("claim:walk-" ^ name) in
  let rec walk m n =
    let reached =
      Rig_support.interrupted
        ~from:(fun _ -> Rig_support.counted () = 0)
        n
        (fun () -> call m)
    in
    let msg = Printf.sprintf "landing at allocation %d" n in
    equal ~msg int 0 (Rig_support.counted ());
    let m = require_some ~msg (Rig_support.donated m) in
    if reached then walk m (n + 1) else n - 1
  in
  greater ~msg:"the landings" int ~than:0
    (walk (B.create ~memory:Mapped d 8) 1)

let test_walk_blit () =
  walk_window "blit" (fun m -> B.blit_from_string "blit" 0 m 2 4)

let test_walk_fill () = walk_window "fill" (fun m -> B.fill m 'f')

(* An exception at any allocation of a take, as an asynchronous one lands,
   leaves both of its buffers live with no claim: the take allocates nothing
   once it has taken the memory. A take no exception ended kills both. *)
let test_walk_take () =
  let rec walk n =
    let a = filled 8 'a' and b = filled 8 'b' in
    let reached =
      Rig_support.interrupted n (fun () ->
          ignore (B.take ~why:"taken" [ a; b ]))
    in
    let msg = Printf.sprintf "landing at allocation %d" n in
    let dead = (B.dead a, B.dead b) in
    if reached then begin
      equal ~msg (pair (option string) (option string)) (None, None) dead;
      is_some ~msg (B.take ~why:"again" [ a; b ]);
      walk (n + 1)
    end
    else begin
      equal ~msg
        (pair (option string) (option string))
        (Some "taken", Some "taken") dead;
      n - 1
    end
  in
  greater ~msg:"the landings" int ~than:0 (walk 1)

(* Claims of several buffers from C *)

(* What a buffer of a case is to [rig_buffers_claim]: live, read or written;
   with a write it must wait for; dead; or admitting only reads, claimed to
   write. *)
type state = Live of Rig.access | Pending | Dead | Read_only

let state =
  Gen.of_list
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | Live Rig.Read -> "read"
        | Live Rig.Read_write -> "written"
        | Pending -> "pending"
        | Dead -> "dead"
        | Read_only -> "read-only"))
    [ Live Rig.Read; Live Rig.Read_write; Pending; Dead; Read_only ]

let states = Gen.list ~size:(Gen.int_range 0 6) state

let refusal = function
  | Dead -> Some R.Dead
  | Read_only -> Some R.Read_only
  | Live _ | Pending -> None

let claims_device = lazy (P.open_ "claim:claim-all")
let read_only = lazy (Rig_support.io "claim:claim-all-read-only")

(* A buffer in [s] and the access it is claimed for. *)
let buffer_in s =
  let d, _ = Lazy.force claims_device in
  match s with
  | Live a -> (B.create Rig.host 8, a)
  | Pending ->
      let b = B.create d 8 in
      write d b;
      (b, Rig.Read)
  | Dead ->
      let b = B.create Rig.host 8 in
      ignore (taken b);
      (b, Rig.Read)
  | Read_only ->
      ( Rig_support.io_buffer ~access:Rig.Read (Lazy.force read_only) 8,
        Rig.Read_write )

(* [rig_buffers_claim] claims every buffer or none: it refuses at the first
   buffer that [rig_buffer_claim] would refuse, holding nothing, and otherwise
   answers each buffer as [rig_buffer_claim] would and holds them all until
   [rig_buffers_release]. *)
let claim_all_law ss =
  let bs = List.map buffer_in ss in
  let buffers = Array.of_list (List.map fst bs) in
  let access = Array.of_list (List.map snd bs) in
  let expected =
    List.find_mapi
      (fun i s -> Option.map (fun a -> (i, a)) (refusal s))
      ss
  in
  (match (R.claim_all buffers access, expected) with
  | R.Refused (i, a), Some (i', a') ->
      cover "a refusal after a claim" (i > 0);
      equal ~msg:"the refused buffer" (pair int answer) (i', a') (i, a)
  | R.All answers, None ->
      cover "every buffer of several claimed" (Array.length answers > 1);
      equal ~msg:"the answers" (array answer)
        (Array.of_list
           (List.map (function Pending -> R.Wait | _ -> R.Claimed) ss))
        answers;
      List.iter
        (fun (b, _) -> equal ~msg:"claimed" bool true (claimed b))
        bs;
      R.release_all buffers
  | R.All _, Some _ -> fail "every buffer claimed past a refusal"
  | R.Refused (i, a), None ->
      failf "buffer %d refused (%a) with none to refuse" i R.pp_answer a);
  List.iter2
    (fun s (b, _) ->
      match s with
      | Live _ | Pending -> equal ~msg:"no claim left" bool true (takes b)
      | Dead | Read_only -> ())
    ss bs;
  ignore (P.run (snd (Lazy.force claims_device)))

(* Keeps *)

(* After the keeps of a memory are unreachable, a call that moves collected
   memory ends them: a take then succeeds. *)
let collect_keeps () =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create Rig.host 8))

(* A keep holds its memory live through takes and frees, which change
   nothing, while reads and writes go on. Once the keep is collected, the
   first call that moves collected memory, on any device, ends it. *)
let test_keep () =
  let d, _ = P.open_ "claim:keep" in
  let a = B.create Rig.host 64 and b = B.create d 64 in
  let k = B.keep [ a; b ] in
  is_none ~msg:"a take of both" (B.take ~why:"taken" [ a; b ]);
  is_none ~msg:"a take of one" (B.take ~why:"taken" [ a ]);
  equal ~msg:"a free" bool false (B.free a);
  equal ~msg:"live" bool true (live [ a; b ]);
  B.fill a 'h';
  equal ~msg:"written under the keep" char 'h'
    (B.read a Bigarray.char (fun x -> x.{0}));
  ignore (Sys.opaque_identity k);
  collect_keeps ();
  is_some ~msg:"a take once collected" (B.take ~why:"taken" [ a; b ])

(* A release ends a keep at once: while the keep is still reachable and
   nothing was collected, a take succeeds. *)
let test_release () =
  let d, _ = P.open_ "claim:release" in
  let a = B.create Rig.host 64 and b = B.create d 64 in
  let k = B.keep [ a; b ] in
  B.release k;
  is_some ~msg:"a take" (B.take ~why:"taken" [ a; b ]);
  ignore (Sys.opaque_identity k)

(* A later release of a keep, and its collection, end nothing: another keep
   of the same memory holds it until its own release. *)
let test_release_once () =
  let a = B.create Rig.host 64 in
  let k1 = B.keep [ a ] and k2 = B.keep [ a ] in
  B.release k1;
  B.release k1;
  is_none ~msg:"after a second release" (B.take ~why:"taken" [ a ]);
  ignore (Sys.opaque_identity k1);
  collect_keeps ();
  is_none ~msg:"after the collection" (B.take ~why:"taken" [ a ]);
  B.release k2;
  B.release (B.keep []);
  ignore (Sys.opaque_identity k2);
  equal ~msg:"after the other's release" bool true (takes a)

(* Two domains: releases of one keep of two memories, beside drops of the
   keep, which release it, collections and takes of each memory. Another keep
   holds the second memory to the end. A release after the first ends
   nothing, so the first memory is taken only once the keep is released, and
   the second never is. A drop beside a release releases the keep twice on
   two domains, and a collection after it runs the drain of the keep's record
   beside the other domain's release. *)
type releases = { mutable released : bool; mutable took : bool }

type kept_pair = {
  first : B.t;
  second : B.t;
  keep : B.keep option Atomic.t;
  keeper : B.keep;
}

let kept_pair =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "released %b, taken %b" r.released r.took)
    ~release:(fun p -> ignore (Sys.opaque_identity p.keeper))
    "p"

let make_pair () =
  let first = B.create Rig.host 64 and second = B.create Rig.host 64 in
  let keep = Atomic.make (Some (B.keep [ first; second ])) in
  { first; second; keep; keeper = B.keep [ second ] }

let release_pair p = Option.iter B.release (Atomic.get p.keep)

(* A drop releases the keep before it empties the cell, so a release that
   finds the cell empty follows the drop's. *)
let drop_pair p =
  match Atomic.get p.keep with
  | None -> ()
  | Some k ->
      B.release k;
      Atomic.set p.keep None

(* A collection that drains: a minor collection finalises a token dropped
   while young, and the allocation drains the list of holds, on which a
   release or the finaliser put the keep's record. *)
let collect_pair (_ : kept_pair) =
  Gc.minor ();
  ignore (Sys.opaque_identity (B.create Rig.host 8))

let release_model r = r.released <- true

(* A take of the first memory succeeds only once the keep is released. It
   may be refused after a release beside another domain's: of two releases
   of one keep, one ends each hold, and the other can return before it
   has. *)
let judge_take_first r = function
  | Ok true ->
      equal ~msg:"released, taken" (pair bool bool) (true, false)
        (r.released, r.took);
      r.took <- true
  | Ok false -> equal ~msg:"taken" bool false r.took
  | Error (Invalid_argument _) -> equal ~msg:"taken" bool true r.took
  | Error e -> raise e

let release_commands =
  [
    command "make"
      (Gen.unit @-> makes kept_pair)
      (fun () -> { released = false; took = false })
      make_pair;
    command "release" (kept_pair ^-> returns unit) release_model release_pair;
    command "drop" (kept_pair ^-> returns unit) release_model drop_pair;
    command "collect" (kept_pair ^-> returns unit) ignore collect_pair;
    command "take"
      (kept_pair ^-> judges bool)
      judge_take_first
      (fun p -> Option.is_some (B.take ~why:"taken" [ p.first ]));
    command "take kept"
      (kept_pair ^-> returns bool)
      (fun _ -> false)
      (fun p -> Option.is_some (B.take ~why:"taken" [ p.second ]));
  ]

(* A submission keeps its fixed memory and the buffers its parts name, which
   every run reads, as a keep does: none of them is taken while the
   submission is reachable, and each is once it is collected. Slots keep
   nothing. *)
let test_submission_keeps () =
  let d, _ = P.open_ "claim:submission-keeps" in
  let fixed = B.create d 8 and src = B.create d 8 and dst = B.create d 8 in
  let ws = B.create Rig.host 8 and slot = B.create d 8 in
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
    (List.map takes [ fixed; src; dst; ws; slot ]);
  ignore (Sys.opaque_identity (submit s ~buffers:[| B.create d 8 |]));
  collect_keeps ();
  equal ~msg:"once it is collected" (list bool) [ true; true; true; true ]
    (List.map takes [ fixed; src; dst; ws ])

(* A keep of a dead buffer raises and keeps none of the buffers it was
   given. *)
let test_keep_refusals () =
  let a = B.create Rig.host 8 and b = B.create Rig.host 8 in
  ignore (taken b);
  raises_match ~msg:"dead" (dead "donated") (fun () -> B.keep [ a; b ]);
  equal ~msg:"the other buffer" bool true (takes a)

(* Two domains: keeps of a memory against takes of it. A take succeeds only
   while no keep holds the memory, and kills it; a keep of a taken memory
   raises. *)
type kept = { mutable taken : bool; mutable keeps : int }

let kept_memory =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "taken %b, keeps %d" r.taken r.keeps)
    "m"

(* The keeps made, reachable to the end of the run. *)
let keeps : B.keep list Atomic.t = Atomic.make []

let rec retain k =
  let ks = Atomic.get keeps in
  if not (Atomic.compare_and_set keeps ks (k :: ks)) then retain k

let keep_one b = retain (B.keep [ b ])

let judge_keep r = function
  | Ok () ->
      equal ~msg:"taken" bool false r.taken;
      r.keeps <- r.keeps + 1
  | Error (Invalid_argument _) -> equal ~msg:"taken" bool true r.taken
  | Error e -> raise e

let judge_take_kept r = function
  | Ok true ->
      equal ~msg:"taken, keeps" (pair bool int) (false, 0) (r.taken, r.keeps);
      r.taken <- true
  | Ok false ->
      equal ~msg:"taken" bool false r.taken;
      greater ~msg:"keeps" int ~than:0 r.keeps
  | Error (Invalid_argument _) -> equal ~msg:"taken" bool true r.taken
  | Error e -> raise e

let keep_commands =
  [
    command "make"
      (Gen.unit @-> makes kept_memory)
      (fun () -> { taken = false; keeps = 0 })
      (fun () -> B.create Rig.host 64);
    command "keep" (kept_memory ^-> judges unit) judge_keep keep_one;
    command "take"
      (kept_memory ^-> judges bool)
      judge_take_kept
      (fun b -> Option.is_some (B.take ~why:"taken" [ b ]));
  ]

(* Claims from C *)

(* A claim from C is a read claim: it sits beside other readers, refuses a
   take until it is released, and is refused on a dead buffer, where it
   leaves no claim behind. The buffer a take gives is claimed at once. *)
let test_c_claims () =
  let b = B.create Rig.host 16 in
  equal ~msg:"for reading" answer R.Claimed (R.claim b Rig.Read);
  equal ~msg:"for writing" answer R.Claimed (R.claim b Rig.Read_write);
  is_none ~msg:"a take under two claims" (B.take ~why:"taken" [ b ]);
  R.release b;
  R.release b;
  let b' = taken b in
  equal ~msg:"the taken buffer" answer R.Claimed (R.claim b' Rig.Read);
  R.release b';
  equal ~msg:"dead" answer R.Dead (R.claim b Rig.Read);
  equal ~msg:"its reason" (option string) (Some "donated") (R.why b);
  equal ~msg:"no claim left" bool true (takes b')

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
      is_none ~msg:"a take while the wait is blocked"
        (B.take ~why:"taken" [ m ]));
  equal ~msg:"the write is done" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  is_none ~msg:"a take under the claim" (B.take ~why:"taken" [ m ]);
  claim_release m;
  is_some ~msg:"a take once released" (B.take ~why:"taken" [ m ])

(* A read whose wait raises leaves no claim: a signal handler's exception
   during the wait. *)
let test_wait_raises () =
  let d, p = P.open_ "claim:wait-raises" in
  let m = B.create d 64 in
  write d m;
  P.interrupt p;
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
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

(* Domains *)

(* Hold submit at a foreign wait, after it has collected the source's uses. A
   take must not overwrite the source before that submit publishes its own
   use, whether the submission names the source as fixed memory or a copy
   part names it. *)
let test_submit_take (name, kind) () =
  let module Sub = Rig.Submission in
  let producer, pp = P.open_ ("claim:submit-producer:" ^ name) in
  let d, p = P.open_ ("claim:submit:" ^ name) in
  let src = B.create d 8 and dst = B.create d 8 in
  Rig_support.store (R.host src) 42;
  Rig_support.store (R.host dst) 0;
  let run, buffers =
    match kind with
    | `Copy ->
        ( Sub.take (Sub.make d [| Sub.(copy ~src:(Slot 0) ~dst:(Slot 1)) |]),
          [| src; dst |] )
    | `Fixed ->
        let s =
          Sub.make
            ~fixed:[ (src, Rig.Read); (dst, Rig.Read_write) ]
            d
            [| Sub.fill Rig_support.carry ~params:24 |]
        in
        let run = Sub.take s and block = Sub.block s 0 in
        Sub.Run.int64 run block 0 (Int64.of_int (Sub.address s dst));
        Sub.Run.int64 run block 8 (Int64.of_int (Sub.address s src));
        Sub.Run.int64 run block 16 8L;
        (* The submit runs on another domain. *)
        Sub.Run.release run;
        (run, [||])
  in
  (* A submission keeps its fixed memory for its life, and claims a run's
     buffers only while a submit runs. *)
  if kind = `Fixed then
    equal ~msg:"the source once the submission is made" bool true
      (claimed src);
  let point = submit (Sub.make producer [||]) in
  P.gate pp;
  let waiting =
    Domain.spawn (fun () -> Rig.submit run ~waits:[| point |] ~buffers)
  in
  let took = ref false in
  Fun.protect
    ~finally:(fun () ->
      P.open_gate pp;
      Rig.Point.wait (Domain.join waiting))
    (fun () ->
      Rig_support.await "submit waiting at the producer's gate" (fun () ->
          P.sleepers pp = 1);
      match Rig_support.donated src with
      | None -> ()
      | Some replacement ->
          took := true;
          B.wait replacement Rig.Read_write;
          Rig_support.store (R.host replacement) 99);
  equal ~msg:"the take and the submitted copy's result" (pair bool int)
    (false, 42)
    (!took, Rig_support.load (R.host dst));
  equal ~msg:"the submitted work completed" int 0 (P.queued p);
  equal ~msg:"the source after hand-over" bool (kind = `Copy) (takes src)

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
          (takes b))
      bs
  in
  let bs = buffers () in
  ignore (submit s ~buffers:(slots bs));
  check bs;
  for i = 0 to 4 do
    let bs = buffers () in
    let replacement = taken ~why:"replaced" bs.(i) in
    raises_match (dead "replaced") (fun () -> submit s ~buffers:(slots bs));
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
  let run = Sub.take s in
  let point = submit (Sub.make producer [||]) in
  P.interrupt pp;
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      raises Exit (fun () ->
          Rig.submit run ~waits:[| point |] ~buffers:[| a; b; a |]));
  let a = Rig_support.donated a and b = Rig_support.donated b in
  equal ~msg:"both claims released" (pair bool bool) (true, true)
    (Option.is_some a, Option.is_some b);
  let a = require_some a and b = require_some b in
  ignore (P.run pp);
  ignore (Rig.submit run ~waits:[| point |] ~buffers:[| a; b; a |]);
  ignore (P.run p)

(* Two domains: takes of a group of two memories, in either order, against a
   read of the second that ends in the same call. Whatever the order, no
   claim outlives its call: at the end the group is taken unless a take took
   it. A take beside a read may find the read's claim and answer [None],
   which no order of whole calls shows: a read's claim spans its call. *)
type grouped = { mutable both : bool }
type group = { b1 : B.t; b2 : B.t }

let end_group g =
  if B.dead g.b1 = None then
    is_some ~msg:"taken at the end" (B.take ~why:"ended" [ g.b1; g.b2 ])

let two =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "taken %b" r.both)
    ~release:end_group "g"

let make_group () = { b1 = B.create Rig.host 64; b2 = B.create Rig.host 64 }
let take_group g = Option.is_some (B.take ~why:"taken" [ g.b1; g.b2 ])

(* A take in the other order, which meets the first's step at its second
   word, holding one, as two such takes on two domains do. *)
let take_reversed g = Option.is_some (B.take ~why:"taken" [ g.b2; g.b1 ])

let read_group g =
  claim_read g.b2;
  claim_release g.b2

let judge_take_group r = function
  | Ok taken ->
      equal ~msg:"taken" bool false r.both;
      r.both <- taken
  | Error (Invalid_argument _) -> equal ~msg:"taken" bool true r.both
  | Error e -> raise e

let judge_read_group r = function
  | Ok () -> equal ~msg:"taken" bool false r.both
  | Error (Invalid_argument _) -> equal ~msg:"taken" bool true r.both
  | Error e -> raise e

let group_commands =
  [
    command "make" (Gen.unit @-> makes two) (fun () -> { both = false })
      make_group;
    command "take" (two ^-> judges bool) judge_take_group take_group;
    command "take reversed" (two ^-> judges bool) judge_take_group
      take_reversed;
    command "read" (two ^-> judges unit) judge_read_group read_group;
  ]

(* Two domains: reads, claims from C, exports and shares against a take that
   writes the memory in place through the buffer it gives, or a free, which
   takes the memory as a take does. A read, an export or a share before the
   take refuses it; one after it finds the buffer dead. *)
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

(* A take and a free end the memory's buffer alike. *)
let judge_take r = function
  | Ok true ->
      equal ~msg:"dead, reads, outside" (triple bool int bool)
        (false, 0, false) (r.dead, r.reads, r.out);
      r.dead <- true
  | Ok false ->
      equal ~msg:"dead" bool false r.dead;
      equal ~msg:"read or outside" bool true (r.reads > 0 || r.out)
  | Error (Invalid_argument _) -> equal ~msg:"dead" bool true r.dead
  | Error e -> raise e

let judge_c_claim r = function
  | Ok R.Claimed ->
      equal ~msg:"dead" bool false r.dead;
      r.reads <- r.reads + 1
  | Ok R.Dead -> equal ~msg:"dead" bool true r.dead
  | Ok a -> failf "claim answered %a" R.pp_answer a
  | Error e -> raise e

let export b = ignore (B.export b Bigarray.char)

let take_write b =
  match Rig_support.donated b with
  | None -> false
  | Some b' ->
      B.write b' Bigarray.char (fun a -> Bigarray.Array1.fill a 'd');
      true

let take_commands =
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
    command "share" (memory ^-> judges unit) judge_outside B.share;
    command "take" (memory ^-> judges bool) judge_take take_write;
    command "free" (memory ^-> judges bool) judge_take B.free;
  ]

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

(* A copy, blit or fill the host runs claims its memory before it waits. In
   another domain, [access] waits at [p]'s gate for work on memory of [p]'s
   device; meanwhile [b] is taken, and a take that succeeds gives [taker]
   the buffer to write in place. It is that buffer if the take succeeded,
   and how [access] ended. *)
let take_while_waiting p b access taker =
  P.gate p;
  let ended =
    Domain.spawn (fun () ->
        match access () with
        | () -> "returned"
        | exception e -> Printexc.to_string e)
  in
  Rig_support.await "the host access waiting at the gate" (fun () ->
      P.sleepers p = 1);
  let got = Rig_support.donated b in
  Option.iter taker got;
  P.open_gate p;
  (got, Domain.join ended)

(* The copy reads the source after the wait for its destination's last use. *)
let test_host_copy_source () =
  let d, p = P.open_ "claim:host-copy-source" in
  let src = filled 64 's' and dst = B.create d 64 in
  write d dst;
  let got, ended =
    take_while_waiting p src
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  equal ~msg:"the copy" string "returned" ended;
  equal ~msg:"taken during the copy" bool false (Option.is_some got);
  equal ~msg:"the copy's bytes" string (String.make 64 's') (text dst)

(* The copy writes the destination after the wait for its source's last
   write. *)
let test_host_copy_destination () =
  let d, p = P.open_ "claim:host-copy-destination" in
  let src = B.create d 64 in
  B.fill src 's';
  write d src;
  let dst = filled 64 'd' in
  let got, ended =
    take_while_waiting p dst
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  Option.iter
    (fun b ->
      equal ~msg:"the taker's bytes" string (String.make 64 'X') (text b))
    got;
  equal ~msg:"the copy" string "returned" ended;
  equal ~msg:"taken during the copy" bool false (Option.is_some got);
  equal ~msg:"the copy's bytes" string (String.make 64 's') (text dst)

(* A blit or fill on memory of [d] the host addresses waits for the work on
   it, and refuses a take meanwhile. *)
let host_accesses =
  [
    ("a blit to bytes", fun b -> B.blit_to_bytes b 0 (Bytes.create 64) 0 64);
    ( "a blit from a string",
      fun b -> B.blit_from_string (String.make 64 'a') 0 b 0 64 );
    ("a fill", fun b -> B.fill b 'f');
  ]

let test_host_access_claims () =
  List.iter
    (fun (name, access) ->
      let d, p = P.open_ ("claim:host-access-" ^ name) in
      let b = B.create d 64 in
      write d b;
      let got, ended = take_while_waiting p b (fun () -> access b) ignore in
      equal ~msg:(name ^ " returned") string "returned" ended;
      equal ~msg:(name ^ ", taken during it") bool false (Option.is_some got))
    host_accesses

(* Two domains: host copies, blits and fills of a memory beside takes of it.
   A take that succeeds fills the buffer it gets and reads it back. Every
   access writes or reads the byte the memory was made with, so while the
   memory is live they agree whatever their order. An access after a take
   finds the buffer dead. A take may also find the memory read by an access
   beside it and answer [None], in any state, which no order of whole calls
   shows: an access's claim spans its call. *)
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

let judge_taker r = function
  | Ok None -> equal ~msg:"dead" bool false r.gone
  | Ok (Some s) ->
      equal ~msg:"dead" bool false r.gone;
      equal ~msg:"the taker's bytes" string (String.make 64 'X') s;
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
    command "take"
      (copied ^-> judges (option string))
      judge_taker
      (fun b ->
        Rig_support.donated b
        |> Option.map (fun b' ->
               B.fill b' 'X';
               text b'));
  ]

let tests =
  [
    group ~timeout "takes"
      [
        prop ~count:law_cases
          "a take takes the memory of every buffer or of none" case take_law;
        test "a take is refused by any claim and leaves its buffers live"
          test_take_refused;
        test "a take inside a read of one of its buffers is refused"
          test_take_reading;
        test "buffers of a take over one memory refuse it" test_take_overlap;
        test "a take takes every memory and kills every earlier buffer"
          test_take;
        test "memories with no address are told apart" test_no_address;
        test "a taker uses the memory through the buffer the take gives"
          test_taken_writes;
        test "a borrow and the memory it maps share one count"
          test_borrow_counts;
        test "a borrow of a taken buffer raises" test_borrow_taken;
        test "a bigarray's memory is never taken" test_of_bigarray;
        test "memory shared through any of its buffers is never taken"
          test_share_covers;
        test "a take of memories of two devices takes both or neither"
          test_shards;
        test "an export of taken memory is the taker's" test_export_taken;
        test "a taker submits work on the buffer the take gives"
          test_taker_submits;
        cases "the buffer a take gives is used as any other" ~id:taken_name
          taken_cases taken_law;
      ];
    group ~timeout "taken memory"
      [
        test "memory taken twice lives while its last buffer does"
          test_take_twice;
        test "a dead buffer refuses claims and keeps" test_dead_claims;
        test "the C claim of a closed device's buffer answers its loss"
          test_closed_slot_claim;
        test "sharing a dead buffer is refused" test_share_refused;
        test "a buffer's death is a fact with its reason" test_dead_fact;
        test "a take of memory a loss reaches raises and takes nothing"
          test_lost_take;
        test "a signal pending as a read's function returns leaves no claim"
          test_signal_returned;
        test "a signal pending as a read's function raises leaves no claim"
          test_signal_raised;
        test "a signal pending as a Mapped write's function returns lands \
              after its flush"
          test_signal_window_returned;
        test "a signal pending as a Mapped write's function raises lands \
              after its flush"
          test_signal_window_raised;
        test "an exception at any allocation of a blit into Mapped memory \
              leaves no claim"
          test_walk_blit;
        test "an exception at any allocation of a fill of Mapped memory \
              leaves no claim"
          test_walk_fill;
        test "an exception at any allocation of a take takes all or none"
          test_walk_take;
      ];
    group ~timeout "keeps"
      [
        test "a keep holds memory from takes until it is collected" test_keep;
        test "a keep of a dead buffer keeps nothing" test_keep_refusals;
        test "a release ends a keep before its collection" test_release;
        test "a later release and the collection of a released keep end \
              nothing"
          test_release_once;
        stateful ~domains:2
          "releases of one keep on two domains beside its collection end it \
           once"
          release_commands;
        test "a submission keeps the memory every run reads"
          test_submission_keeps;
        stateful ~domains:2
          "keeps beside takes of one memory are ordered with them"
          keep_commands;
      ];
    group ~timeout "claims from C"
      [
        prop ~count:law_cases
          "rig_buffers_claim claims every buffer or none" states claim_all_law;
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
        test "fixed memory refuses a take during submit"
          (test_submit_take ("fixed", `Fixed));
        test "a run's buffers refuse a take during submit"
          (test_submit_take ("copy", `Copy));
        test "a run's refused buffer releases every claim the submit took"
          test_run_claim_refusals;
        test "an exceptional submit wait releases its run's claims"
          test_run_wait_raises;
        stateful ~domains:2
          "takes of a group and reads of its memory leave no claim behind"
          group_commands;
        stateful ~domains:2
          "reads, exports and shares beside a take or a free are ordered \
           with it"
          take_commands;
        test "a wait beside a free walks the stamps it began on"
          test_free_beside_wait;
        test "a free beside a borrow that maps changes nothing"
          test_free_beside_borrow;
        test "a host copy waiting holds its source from a take"
          test_host_copy_source;
        test "a host copy waiting holds its destination from a take"
          test_host_copy_destination;
        test "a host blit or fill waiting holds its memory from a take"
          test_host_access_claims;
        stateful ~domains:2
          "host copies, blits and fills beside takes are ordered with them"
          copy_commands;
      ];
  ]

let () = exit (run "rig.claim" tests)
