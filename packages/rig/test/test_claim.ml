(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
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

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

let write d m =
  let s = Sub.make d [| Rig_support.touching [| Sub.write |] |] in
  ignore (submit ~args:(Rig_support.zeros 1) s ~buffers:[| m |])

let read d m =
  let s = Sub.make d [| Rig_support.touching [| Sub.read |] |] in
  ignore (submit ~args:(Rig_support.zeros 1) s ~buffers:[| m |])

let lost = function Rig.Lost _ -> true | _ -> false
let lost_by d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* Loses [d]: its next submit fails. *)
let lose d p =
  P.fail p;
  try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ()

let answer = Testable.make ~pp:R.pp_answer ~equal:( = )

let access =
  Testable.make
    ~pp:(fun ppf a ->
      Format.pp_print_string ppf
        (match a with Rig.Read -> "Read" | Rig.Read_write -> "Read_write"))
    ~equal:( = )

let dead why = Exn.invalid_arg ~sub:why

(* The death of each buffer: [None] while it lives, the take's reason once its
   memory is taken. *)
let deaths bs = List.map B.dead bs

let text b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

let filled ?(on = Rig.host) n c =
  let b = B.create on n in
  B.fill b c;
  b

(* What a take of [bs] does: ["taken"], which kills them, or ["refused"], which
   changes nothing. *)
let take_of ?(why = "taken") bs =
  match B.take ~why bs with Some _ -> "taken" | None -> "refused"

(* A take of [b] alone, with the reason ["donated"]. *)
let alone b = take_of ~why:"donated" [ b ]

(* The buffer a take of [b] alone gives, with the reason [why]. *)
let taken ?(why = "donated") b =
  match B.take ~why [ b ] with
  | Some [ b' ] -> b'
  | Some bs -> failf "a take of one buffer gave %d" (List.length bs)
  | None -> fail "a take of memory nothing claims was refused"

(* Read claims *)

(* The access's function takes [c] and [b], and the take again once the access
   returned. *)
let in_function =
  [
    ("a read's", fun b f -> B.read b Bigarray.char (fun _ -> f ()));
    ("a write's", fun b f -> B.write b Bigarray.char (fun _ -> f ()));
  ]

let test_take_in_function (_, access) =
  let c = filled 8 'c' and b = filled 8 'b' in
  let inside = ref "" in
  access b (fun () -> inside := take_of [ c; b ]);
  equal ~msg:"inside the function" string "refused" !inside;
  equal ~msg:"live" (list (option string)) [ None; None ] (deaths [ c; b ]);
  equal ~msg:"once it returned" string "taken" (take_of [ c; b ])

(* In another domain, [access] waits at [p]'s gate for work on memory of [p]'s
   device; meanwhile [b] is taken, and a take that succeeds gives [taker] the
   buffer to write in place. It is what the take did, and how [access] ended. *)
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
  let took = match got with Some _ -> "taken" | None -> "refused" in
  (took, Domain.join ended)

(* Each waits for the work on memory of a device the host addresses. *)
let host_accesses =
  [
    ("a read", fun b -> B.read b Bigarray.char ignore);
    ("a write", fun b -> B.write b Bigarray.char ignore);
    ("a blit to bytes", fun b -> B.blit_to_bytes b 0 (Bytes.create 64) 0 64);
    ( "a blit from a string",
      fun b -> B.blit_from_string (String.make 64 'a') 0 b 0 64 );
    ("a fill", fun b -> B.fill b 'f');
  ]

let test_host_access_waiting (name, access) =
  let d, p = P.open_ ("claim:host-access-" ^ name) in
  let b = B.create d 64 in
  write d b;
  equal (pair string string) ("refused", "returned")
    (take_while_waiting p b (fun () -> access b) ignore)

(* The copy reads the source after the wait for its destination's last use. *)
let test_host_copy_source () =
  let d, p = P.open_ "claim:host-copy-source" in
  let src = filled 64 's' and dst = B.create d 64 in
  write d dst;
  let took, ended =
    take_while_waiting p src
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  equal ~msg:"the take, the copy" (pair string string) ("refused", "returned")
    (took, ended);
  equal ~msg:"the copy's bytes" string (String.make 64 's') (text dst)

(* The copy writes the destination after the wait for its source's last
   write. *)
let test_host_copy_destination () =
  let d, p = P.open_ "claim:host-copy-destination" in
  let src = B.create d 64 in
  B.fill src 's';
  write d src;
  let dst = filled 64 'd' in
  let took, ended =
    take_while_waiting p dst
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  equal ~msg:"the take, the copy" (pair string string) ("refused", "returned")
    (took, ended);
  equal ~msg:"the copy's bytes" string (String.make 64 's') (text dst)

(* Two domains: host copies, blits and fills of a memory beside takes of it. A
   take that succeeds fills the buffer it gets and reads it back. Every access
   writes or reads the byte the memory was made with, so while the memory is
   live they agree whatever their order. An access after a take finds the buffer
   dead. A take may also find the memory read by an access beside it and answer
   [None], in any state, which no order of whole calls shows: an access's claim
   spans its call. *)
type copied = { mutable gone : bool }

let copied = abstract ~pp:(fun ppf r -> Format.fprintf ppf "dead %b" r.gone) "m"
let made = 's'

(* An access of a live memory returns; of a dead one it raises. *)
let judge_live r = function
  | Ok _ when r.gone -> fail "an access of a taken memory returned"
  | Ok _ -> ()
  | Error (Invalid_argument _) when r.gone -> ()
  | Error e -> raise e

let judge_bytes r outcome =
  judge_live r outcome;
  match outcome with
  | Ok s -> equal ~msg:"bytes" string (String.make 64 made) s
  | Error _ -> ()

let judge_taker r outcome =
  judge_live r outcome;
  match outcome with
  | Ok (Some s) ->
      equal ~msg:"the taker's bytes" string (String.make 64 'X') s;
      r.gone <- true
  | Ok None | Error _ -> ()

let copy_commands =
  [
    command "make"
      (Gen.unit @-> makes copied)
      (fun () -> { gone = false })
      (fun () -> filled 64 made);
    command "copy out"
      (copied ^-> judges string)
      judge_bytes
      (fun b ->
        let dst = B.create Rig.host 64 in
        B.copy ~src:b ~dst;
        text dst);
    command "copy in"
      (copied ^-> judges unit)
      judge_live
      (fun b -> B.copy ~src:(filled 64 made) ~dst:b);
    command "blit out" (copied ^-> judges string) judge_bytes text;
    command "blit in"
      (copied ^-> judges unit)
      judge_live
      (fun b -> B.blit_from_string (String.make 64 made) 0 b 0 64);
    command "fill" (copied ^-> judges unit) judge_live (fun b -> B.fill b made);
    command "take"
      (copied ^-> judges (option string))
      judge_taker
      (fun b ->
        Rig_support.donated b
        |> Option.map (fun b' ->
            B.fill b' 'X';
            text b'));
  ]

(* The take succeeds while the write it follows waits at the gate; the taker's
   host write then waits for that work. *)
let test_device_work () =
  let d, p = P.open_ "claim:device-work" in
  let b = B.create d 64 in
  P.gate p;
  write d b;
  let b' = taken b in
  let wrote = Atomic.make false in
  let writer =
    Domain.spawn (fun () ->
        B.write b' Bigarray.char (fun a -> Bigarray.Array1.fill a 'w');
        Atomic.set wrote true)
  in
  Rig_support.await "the write waiting, or returned" (fun () ->
      P.sleepers p = 1 || Atomic.get wrote);
  let waited = P.sleepers p in
  P.open_gate p;
  Domain.join writer;
  equal ~msg:"writes waiting for the work" int 1 waited;
  equal ~msg:"the write" string (String.make 64 'w') (text b')

(* A read whose wait raises: a signal handler's exception during the wait. *)
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
  equal ~msg:"a take after it" string "taken" (alone m);
  ignore (P.run p)

(* A signal pending as a read's function returns or raises lands after the read
   ended its claim. [f]'s last act raises SIGINT, which the runtime records; its
   handler raises [Exit] at the next poll point, which the allocation after the
   call places if the read has none left. *)
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
  equal ~msg:"a take after it" string "taken" (alone b)

let test_signal_returned () =
  interrupted_at_end (fun _ -> Rig_support.interrupt ())

let test_signal_raised () =
  interrupted_at_end (fun _ ->
      Rig_support.interrupt ();
      raise_notrace Raised)

(* An exception at any allocation of a blit or a fill into Mapped memory outside
   its counted call, as an asynchronous one lands, leaves no claim and no count.
   Each landing's check takes the memory, and the next landing uses the buffer
   the take gave. *)
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
  greater ~msg:"the landings" int ~than:0 (walk (B.create ~memory:Mapped d 8) 1)

let test_walk_blit () =
  walk_window "blit" (fun m -> B.blit_from_string "blit" 0 m 2 4)

let test_walk_fill () = walk_window "fill" (fun m -> B.fill m 'f')

(* Keeps *)

(* After the keeps of a memory are unreachable, a call that moves collected
   memory ends them: a take then succeeds. *)
let collect_keeps () =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create Rig.host 8))

let freed b = if B.free b then "freed" else "kept"

(* Takes and frees change nothing, while reads and writes go on. Once the keep
   is collected, the first call that moves collected memory, on any device, ends
   it. *)
let test_keep () =
  let d, _ = P.open_ "claim:keep" in
  let a = B.create Rig.host 64 and b = B.create d 64 in
  let k = B.keep [ a; b ] in
  equal ~msg:"a take of both" string "refused" (take_of [ a; b ]);
  equal ~msg:"a take of one" string "refused" (take_of [ a ]);
  equal ~msg:"a free" string "kept" (freed a);
  equal ~msg:"live" (list (option string)) [ None; None ] (deaths [ a; b ]);
  B.fill a 'h';
  equal ~msg:"written under the keep" char 'h'
    (B.read a Bigarray.char (fun x -> x.{0}));
  ignore (Sys.opaque_identity k);
  collect_keeps ();
  equal ~msg:"a take once collected" string "taken" (take_of [ a; b ])

let test_keep_refusals () =
  let a = B.create Rig.host 8 and b = B.create Rig.host 8 in
  ignore (taken b);
  raises_match ~msg:"dead" (dead "donated") (fun () -> B.keep [ a; b ]);
  equal ~msg:"the other buffer" string "taken" (alone a)

(* The keep is still reachable and nothing was collected. *)
let test_release () =
  let d, _ = P.open_ "claim:release" in
  let a = B.create Rig.host 64 and b = B.create d 64 in
  let k = B.keep [ a; b ] in
  B.release k;
  equal string "taken" (take_of [ a; b ]);
  ignore (Sys.opaque_identity k)

(* Another keep of the same memory holds it until its own release. *)
let test_release_once () =
  let a = B.create Rig.host 64 in
  let k1 = B.keep [ a ] and k2 = B.keep [ a ] in
  B.release k1;
  B.release k1;
  equal ~msg:"after a second release" string "refused" (take_of [ a ]);
  ignore (Sys.opaque_identity k1);
  collect_keeps ();
  equal ~msg:"after the collection" string "refused" (take_of [ a ]);
  B.release k2;
  B.release (B.keep []);
  ignore (Sys.opaque_identity k2);
  equal ~msg:"after the other's release" string "taken" (alone a)

(* Two domains: one releases a keep of two memories, and its next call's drain,
   which finds the keep's record, is held before its first step
   ({!Rig.on_drain}) while the other domain releases the keep again, takes the
   first memory, and drops and collects the keep. The take succeeds; the drain
   then ends no hold again, so the second memory, which another keep holds, is
   refused until that keep's release. The two releases do not overlap here: the
   stateful below reaches that. *)
let test_releases_beside_drain () =
  let first = B.create Rig.host 64 and second = B.create Rig.host 64 in
  let keeper = B.keep [ second ] in
  let keep = ref (Some (B.keep [ first; second ])) in
  equal ~msg:"a take under the keep" string "refused" (take_of [ first ]);
  let entered = Atomic.make false and go = Atomic.make false in
  Rig.on_drain (fun () ->
      Atomic.set entered true;
      Rig_support.await "the other domain's release" (fun () -> Atomic.get go));
  let held =
    Domain.spawn (fun () ->
        B.release (Option.get !keep);
        ignore (Sys.opaque_identity (B.create Rig.host 8)))
  in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set go true;
      Domain.join held)
    (fun () ->
      Rig_support.await "the held drain" (fun () -> Atomic.get entered);
      B.release (Option.get !keep);
      equal ~msg:"a take after the releases" string "taken" (take_of [ first ]);
      keep := None;
      Gc.full_major ();
      equal ~msg:"the second memory while the drain is held" string "refused"
        (take_of [ second ]));
  equal ~msg:"the second memory once the drain returned" string "refused"
    (take_of [ second ]);
  B.release keeper;
  equal ~msg:"once the other keep is released" string "taken"
    (take_of [ second ])

(* Two domains: releases of one keep of two memories, and takes of each. Another
   keep holds the second memory to the end. Each release returns once the keep's
   holds have ended, also beside another domain's release that is ending them,
   so a take after any release succeeds; the second memory is never taken. That
   overlap lies inside the one C step that ends a hold, which no hook holds: the
   branches reach it by chance, on each of a program's 50 runs. With no
   collection a program costs little. Its rarest cover, "a take after a
   release", marks 27.5% of 12000 cases, at least 26.7%: [~count:70] misses it
   with odds 4e-10. *)
type releases = { mutable released : bool; mutable took : bool }
type kept_pair = { first : B.t; second : B.t; keep : B.keep; keeper : B.keep }

let kept_pair =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "released %b, taken %b" r.released r.took)
    ~release:(fun p -> ignore (Sys.opaque_identity p.keeper))
    "p"

let make_pair () =
  let first = B.create Rig.host 64 and second = B.create Rig.host 64 in
  let keep = B.keep [ first; second ] in
  { first; second; keep; keeper = B.keep [ second ] }

let release_model r =
  cover "a release of a released keep" r.released;
  r.released <- true

let judge_take_first r taken =
  cover "a take after a release" r.released;
  match (taken, r.took, r.released) with
  | Ok "taken", false, true -> r.took <- true
  | Ok "taken", true, _ -> fail "a take of a taken memory succeeded"
  | Ok "taken", false, false -> fail "a take under the keep succeeded"
  | Ok _, false, false -> ()
  | Ok _, false, true -> fail "a take after the release was refused"
  | Ok _, true, _ -> fail "a take of a taken memory returned"
  | Error (Invalid_argument _), true, _ -> ()
  | Error e, _, _ -> raise e

let release_commands =
  [
    command "make"
      (Gen.unit @-> makes kept_pair)
      (fun () -> { released = false; took = false })
      make_pair;
    command "release"
      (kept_pair ^-> returns unit)
      release_model
      (fun p -> B.release p.keep);
    command "take"
      (kept_pair ^-> judges string)
      judge_take_first
      (fun p -> take_of [ p.first ]);
    command "take kept"
      (kept_pair ^-> returns string)
      (fun _ -> "refused")
      (fun p -> take_of [ p.second ]);
  ]

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
  | Ok () when r.taken -> fail "a keep of a taken memory returned"
  | Ok () -> r.keeps <- r.keeps + 1
  | Error (Invalid_argument _) when r.taken -> ()
  | Error e -> raise e

let judge_take_kept r outcome =
  match (outcome, r.taken) with
  | Error (Invalid_argument _), true -> ()
  | Error e, _ -> raise e
  | Ok _, true -> fail "a take of a taken memory returned"
  | Ok "taken", false when r.keeps > 0 -> fail "a take of kept memory succeeded"
  | Ok "taken", false -> r.taken <- true
  | Ok _, false when r.keeps = 0 -> fail "a take of unkept memory was refused"
  | Ok _, false -> ()

let keep_commands =
  [
    command "make"
      (Gen.unit @-> makes kept_memory)
      (fun () -> { taken = false; keeps = 0 })
      (fun () -> B.create Rig.host 64);
    command "keep" (kept_memory ^-> judges unit) judge_keep keep_one;
    command "take"
      (kept_memory ^-> judges string)
      judge_take_kept
      (fun b -> take_of [ b ]);
  ]

(* Outside the claims *)

let test_of_bigarray () =
  let b =
    B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8)
  in
  equal ~msg:"a take" string "refused" (alone b);
  equal ~msg:"its death" (option string) None (B.dead b)

(* A view, a borrow, the buffer itself. Sharing it again changes nothing, and
   readers claim and release it as before. *)
let test_share_covers () =
  let d = Result.get_ok (Rig.software "claim:share") in
  let fresh () =
    let b = B.create Rig.host B.paged_from in
    (b, B.view b ~first:0 ~length:8, require_some (B.borrow d b))
  in
  let b, view, borrowed = fresh () in
  B.share view;
  equal ~msg:"shared through a view, the buffer" string "refused" (alone b);
  equal ~msg:"shared through a view, the borrow" string "refused"
    (alone borrowed);
  B.share view;
  B.share b;
  equal ~msg:"shared again" string "refused" (alone b);
  claim_read b;
  claim_release b;
  let b, _, borrowed = fresh () in
  B.share borrowed;
  equal ~msg:"shared through the borrow" string "refused" (alone b);
  let b, _, borrowed = fresh () in
  B.share b;
  equal ~msg:"shared through the buffer" string "refused" (alone borrowed);
  let b, _, _ = fresh () in
  equal ~msg:"unshared" string "taken" (alone b)

(* The bus address of a PCI function, which Polled's [dma] describes any region
   to, and refuses where told. *)
let fn = "0000:41:00.0"

let test_dma () =
  let d, p = P.open_ "claim:dma" in
  let described = B.create d 64 and refused = B.create d 64 in
  ignore (require_ok ~pp:Format.pp_print_string (B.dma described ~fn));
  P.fail_at p 1 (`Refuse 1);
  ignore (require_error (B.dma refused ~fn));
  equal
    (list (pair string string))
    [ ("described", "refused"); ("refused", "refused") ]
    [ ("described", alone described); ("refused", alone refused) ]

(* The buffer a take gives is shared as any other. *)
let test_share_refused () =
  let b = B.create Rig.host 16 in
  let b' = taken b in
  raises_match ~msg:"dead" (dead "donated") (fun () -> B.share b);
  let b'' = taken ~why:"again" b' in
  B.share b'';
  raises_match ~msg:"dead, its memory shared" (dead "again") (fun () ->
      B.share b');
  equal ~msg:"the live buffer, shared" string "refused" (alone b'')

(* Takes *)

(* Takes of views of several memories, against a model *)

(* What uses a memory before the take: nothing, a read claim, a keep, an export,
   or a share through a view of its last byte. *)
type use = Unused | Read_claim | Kept | Exported | Shared

(* A case: the bytes of each memory, each memory's use, and the take's buffers,
   each a view of a memory: its index, first byte and length. Memories of odd
   index lie on a software device, but for an exported one, since only the
   host's memory exports; the others lie on the host. *)
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
       (List.map (fun (m, o, l) -> Printf.sprintf "[%d: %d,+%d]" m o l) c.views))

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
          (pair
             (int_range 0 (k - 1))
             (of_list [ Read_claim; Kept; Exported; Shared ])))
  in
  (* The first memory named twice. *)
  let twice n =
    map
      (fun k -> { n; uses = unused k; views = (0, 0, n) :: wholes n k })
      (int_range 1 3)
  in
  (* The first memory named by a view of part of it, which a memory of no bytes
     has none of. *)
  let part n =
    if n = 0 then clean n
    else
      bind
        (pair (int_range 1 3) (int_range 0 n))
        (fun (k, o) ->
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
    bind
      (int_range 0 (k - 1))
      (fun m ->
        frequency
          [
            (3, constant (m, 0, n));
            ( 1,
              bind (int_range 0 n) (fun o ->
                  map (fun l -> (m, o, l)) (int_range 0 (n - o))) );
          ])
  in
  let any n =
    bind
      (list ~size:(int_range 1 4) use)
      (fun uses ->
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

(* A take of an unused memory, then of one each use holds: the take must leave
   no claim on the memory it reached first. *)
let after_held =
  List.map
    (fun u -> { n = 8; uses = [ Unused; u ]; views = [ (0, 0, 8); (1, 0, 8) ] })
    [ Read_claim; Kept; Exported; Shared ]

let spans_all c = List.for_all (fun (_, o, l) -> o = 0 && l = c.n) c.views

let named_once c =
  let named = List.map (fun (m, _, _) -> m) c.views in
  List.length (List.sort_uniq Int.compare named) = List.length named

let use_of c m = List.nth c.uses m
let named_use c u = List.exists (fun (m, _, _) -> use_of c m = u) c.views

(* Whether the first view of a used memory follows a view of an unused one. *)
let used_after_unused c =
  let rec go held = function
    | [] -> false
    | (m, _, _) :: rest -> if use_of c m <> Unused then held else go true rest
  in
  go false c.views

(* What a take must do: take every memory when each view spans its memory, no
   memory is named twice, and nothing uses a memory named. *)
let takes_all c =
  spans_all c && named_once c
  && List.for_all (fun (m, _, _) -> use_of c m = Unused) c.views

let second_device =
  fixture (fun () -> Result.get_ok (Rig.software "claim:take-second"))

let take_law c =
  let made i = Char.chr (Char.code 'a' + i) in
  let on i u =
    if i mod 2 = 1 && u <> Exported then second_device () else Rig.host
  in
  let memories =
    List.mapi (fun i u -> filled ~on:(on i u) c.n (made i)) c.uses
  in
  (* A view of each memory's last byte, made before the take. *)
  let lasts =
    List.map
      (fun b -> B.view b ~first:(max 0 (c.n - 1)) ~length:(min c.n 1))
      memories
  in
  (* Starts [u] on [b]'s memory, and is how it ends: an export or a share lasts
     for good. *)
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
  let unused = List.for_all (fun (m, _, _) -> use_of c m = Unused) c.views in
  cover "several memories taken" (e && List.length c.views >= 2);
  cover "memories of two devices taken" (e && named 0 && named 1);
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
  cover "refused by a use of a memory after one the take held"
    (whole && used_after_unused c);
  let got = B.take ~why:"taken" buffers in
  let outcome = match got with Some _ -> "taken" | None -> "refused" in
  equal ~msg:"the take" string (if e then "taken" else "refused") outcome;
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
          let death = if named i then Some "taken" else None in
          let msg = Printf.sprintf "memory %d's buffer, its last byte" i in
          equal ~msg
            (pair (option string) (option string))
            (death, death)
            (B.dead b, B.dead (List.nth lasts i)))
        memories;
      List.iter
        (fun v ->
          equal ~msg:"a view taken" (option string) (Some "taken") (B.dead v))
        buffers;
      equal ~msg:"the buffers given" int (List.length buffers) (List.length bs);
      List.iter2
        (fun (m, _, _) b ->
          let msg = Printf.sprintf "memory %d's new buffer" m in
          equal ~msg string (String.make c.n (made m)) (text b);
          equal ~msg
            (triple (option string) bool access)
            (None, true, Rig.Read_write)
            (B.dead b, B.spans b, B.access b))
        c.views bs);
  List.iter (fun f -> f ()) ends;
  (* A refused take left no claim: each memory still live, its use ended, takes
     alone unless it is outside the claims. *)
  List.iteri
    (fun i b ->
      if B.dead b = None then
        equal
          ~msg:(Printf.sprintf "memory %d taken alone after" i)
          string
          (match use_of c i with
          | Exported | Shared -> "refused"
          | Unused | Read_claim | Kept -> "taken")
          (alone b))
    memories

let test_read_only () =
  let io = Rig_support.io "claim:take-read-only" in
  let c = filled 8 'c' and b = Rig_support.io_buffer ~access:Rig.Read io 8 in
  equal ~msg:"the take" string "refused" (take_of [ c; b ]);
  equal ~msg:"live" (list (option string)) [ None; None ] (deaths [ c; b ]);
  equal ~msg:"c's bytes" string "cccccccc" (text c);
  equal ~msg:"a take of c alone after it" string "taken" (alone c)

(* The second buffer is the first itself, or a borrow of its memory. *)
let test_take_overlap () =
  let d = Result.get_ok (Rig.software "claim:take-overlap") in
  let c = filled 8 'c' and b = B.create Rig.host B.paged_from in
  List.iter
    (fun (msg, b') ->
      equal ~msg string "refused" (take_of [ c; b; b' ]);
      equal ~msg:(msg ^ ": live")
        (list (option string))
        [ None; None; None ]
        (deaths [ c; b; b' ]))
    [ ("the buffer twice", b); ("a borrow of it", require_some (B.borrow d b)) ]

(* Memories of one device that has no address of them are told apart all the
   same: two are taken together, one twice is refused. *)
let test_no_address () =
  let d, _ = P.open_ ~addresses:false ~host_visible:false "claim:no-address" in
  if Rig.reaches Rig.host d then fail "the host addresses the device's memory";
  let a = B.create d 64 and b = B.create d 64 in
  equal ~msg:"one twice" string "refused"
    (take_of [ a; B.view a ~first:0 ~length:64 ]);
  equal ~msg:"both" string "taken" (take_of [ a; b ])

(* The host's reads and writes through an earlier buffer and every copy, blit or
   fill of one, whether the host, a device or staging memory runs it, raise with
   the take's reason, and each of them runs on the new buffer. *)
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

(* A claim through the borrow refuses a take of the memory until it ends. *)
let test_borrow_counts () =
  let d = Result.get_ok (Rig.software "claim:borrow") in
  let b = B.create Rig.host B.paged_from in
  let borrowed = require_some (B.borrow d b) in
  claim_read borrowed;
  equal ~msg:"under the claim" string "refused" (take_of [ b ]);
  claim_release borrowed;
  equal ~msg:"once it ended" string "taken" (take_of [ b ])

let test_borrow_taken () =
  let d = Result.get_ok (Rig.software "claim:borrow-taken") in
  let b = B.create Rig.host B.paged_from in
  let b' = taken b in
  raises_match (dead "donated") (fun () -> B.borrow d b);
  ignore (require_some ~msg:"the taken buffer's borrow" (B.borrow d b'))

let test_shards () =
  let d = Result.get_ok (Rig.software "claim:shards") in
  let a = B.create Rig.host 8 and b = B.create d 8 in
  claim_read b;
  equal ~msg:"one claimed" string "refused" (take_of [ a; b ]);
  equal ~msg:"both live" (list (option string)) [ None; None ] (deaths [ a; b ]);
  claim_release b;
  equal ~msg:"neither claimed" string "taken" (take_of [ a; b ])

let test_export_taken () =
  let b = B.create Rig.host 16 in
  let b' = taken b in
  raises_match ~msg:"the earlier buffer" (dead "donated") (fun () ->
      B.export b Bigarray.char);
  Bigarray.Array1.fill (B.export b' Bigarray.char) 'x';
  equal ~msg:"its bytes" string (String.make 16 'x') (text b');
  equal ~msg:"a take after the export" string "refused" (alone b')

(* The work writes the memory in place, and the host reads the write. *)
let test_taker_submits () =
  let d, _ = P.open_ "claim:taker-submits" in
  let b = B.create d 8 in
  Rig_support.store (R.host b) 0;
  let b' = taken b in
  let s =
    Sub.make
      ~fixed:[ (b', Rig.Read_write) ]
      d
      [| Sub.fill Rig_support.poke ~params:16 |]
  in
  let params = Bytes.create 16 in
  Bytes.set_int64_le params 0 (Int64.of_int (Sub.address s b'));
  Bytes.set_int64_le params 8 42L;
  let args = [| { Rig_support.zero with params } |] in
  Rig.Point.wait (Rig.submit s ~buffers:[||] ~args);
  equal ~msg:"the word the work stored" int 42
    (B.read b' Bigarray.int64 (fun a -> Int64.to_int a.{0}))

type place = Host | Addressed | Unaddressed

let pp_place ppf = function
  | Host -> Format.pp_print_string ppf "host"
  | Addressed -> Format.pp_print_string ppf "addressed"
  | Unaddressed -> Format.pp_print_string ppf "unaddressed"

let addressed = fixture (fun () -> fst (P.open_ "claim:taken-addressed"))

let unaddressed =
  fixture (fun () ->
      fst (P.open_ ~host_visible:false "claim:taken-unaddressed"))

let device = function
  | Host -> Rig.host
  | Addressed -> addressed ()
  | Unaddressed -> unaddressed ()

let taken_cases =
  List.concat_map
    (fun place -> List.map (fun n -> (place, n)) [ 0; 1; 8; 4096 ])
    [ Host; Addressed; Unaddressed ]

let taken_name (place, n) =
  Format.asprintf "%a memory of %d bytes" pp_place place n

(* Work submitted on it, the host's reads and writes, a copy a device runs, a
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
  equal ~msg:"a take after the share" string "refused" (alone b')

(* Every earlier buffer is dead. *)
let test_take_twice () =
  let b = B.create Rig.host B.paged_from in
  let view = B.view b ~first:0 ~length:8 in
  let b' = taken ~why:"first" b in
  let b'' = taken ~why:"second" b' in
  raises_match Exn.invalid_arg (fun () -> B.wait b Rig.Read);
  raises_match Exn.invalid_arg (fun () -> B.wait view Rig.Read);
  raises_match (dead "second") (fun () -> B.wait b' Rig.Read);
  B.write b'' Bigarray.char (fun a -> Bigarray.Array1.fill a 'c');
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create Rig.host B.paged_from));
  equal char 'c' (B.read b'' Bigarray.char (fun a -> a.{B.paged_from - 1}))

(* The buffer the take gave is claimed. *)
let test_dead_claims () =
  let b = B.create Rig.host 8 in
  let live = taken b in
  equal ~msg:"a claim from C" answer R.Dead (R.claim b Rig.Read);
  raises_match ~msg:"a read" (dead "donated") (fun () ->
      B.read b Bigarray.char ignore);
  raises_match ~msg:"a keep" (dead "donated") (fun () -> B.keep [ b ]);
  claim_read live;
  claim_release live

(* Wherever the buffer lies in the list. A claim on another buffer of the take
   makes it [None] instead, in either order: it would take nothing. *)
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
  equal ~msg:"before a claimed buffer" string "refused" (take_of [ m; h ]);
  equal ~msg:"after a claimed buffer" string "refused" (take_of [ h; m ]);
  claim_release h;
  equal ~msg:"the live buffer" string "taken" (alone h)

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
    let dead = deaths [ a; b ] in
    if reached then begin
      equal ~msg (list (option string)) [ None; None ] dead;
      equal ~msg string "taken" (take_of ~why:"again" [ a; b ]);
      walk (n + 1)
    end
    else begin
      equal ~msg (list (option string)) [ Some "taken"; Some "taken" ] dead;
      n - 1
    end
  in
  greater ~msg:"the landings" int ~than:0 (walk 1)

(* Two domains: takes of a group of two memories, in either order, against a
   read of the second that ends in the same call. Whatever the order, no claim
   outlives its call: at the end the group is taken unless a take took it. A
   take beside a read may find the read's claim and answer [None], which no
   order of whole calls shows: a read's claim spans its call. *)
type grouped = { mutable both : bool }
type group = { b1 : B.t; b2 : B.t }

let end_group g =
  if B.dead g.b1 = None then
    equal ~msg:"a take at the end" string "taken"
      (take_of ~why:"ended" [ g.b1; g.b2 ])

let two =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "taken %b" r.both)
    ~release:end_group "g"

let make_group () = { b1 = B.create Rig.host 64; b2 = B.create Rig.host 64 }
let take_group g = take_of [ g.b1; g.b2 ]

(* A take in the other order, which meets the first's step at its second word,
   holding one, as two such takes on two domains do. *)
let take_reversed g = take_of [ g.b2; g.b1 ]

let read_group g =
  claim_read g.b2;
  claim_release g.b2

(* A call on a taken group raises; on a live one it returns. *)
let judge_group r = function
  | Ok _ when r.both -> fail "a call on a taken group returned"
  | Ok _ -> ()
  | Error (Invalid_argument _) when r.both -> ()
  | Error e -> raise e

let judge_take_group r outcome =
  judge_group r outcome;
  if outcome = Ok "taken" then r.both <- true

let group_commands =
  [
    command "make"
      (Gen.unit @-> makes two)
      (fun () -> { both = false })
      make_group;
    command "take" (two ^-> judges string) judge_take_group take_group;
    command "take reversed"
      (two ^-> judges string)
      judge_take_group take_reversed;
    command "read" (two ^-> judges unit) judge_group read_group;
  ]

(* Two domains: reads, claims from C, exports and shares against a take that
   writes the memory in place through the buffer it gives, or a free, which
   takes the memory as a take does. A read, an export or a share before the take
   refuses it; one after it finds the buffer dead. *)
type memory = { mutable dead : bool; mutable reads : int; mutable out : bool }

let memory =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "dead %b, reads %d, outside %b" r.dead r.reads r.out)
    "m"

(* A use of a dead memory raises; of a live one it returns. *)
let judge_use r = function
  | Ok _ when r.dead -> fail "a use of a dead memory returned"
  | Ok _ -> ()
  | Error (Invalid_argument _) when r.dead -> ()
  | Error e -> raise e

let judge_read r outcome =
  judge_use r outcome;
  if Result.is_ok outcome then r.reads <- r.reads + 1

let judge_outside r outcome =
  judge_use r outcome;
  if Result.is_ok outcome then r.out <- true

(* A take and a free end the memory's buffer alike. *)
let judge_take r outcome =
  judge_use r outcome;
  match outcome with
  | Ok "ended" when r.reads > 0 || r.out ->
      fail "a take of read or outside memory succeeded"
  | Ok "ended" -> r.dead <- true
  | Ok _ when r.reads = 0 && not r.out ->
      fail "a take of memory nothing claims was refused"
  | Ok _ | Error _ -> ()

let judge_c_claim r = function
  | Ok R.Claimed when r.dead -> fail "a claim of a dead memory answered Claimed"
  | Ok R.Claimed -> r.reads <- r.reads + 1
  | Ok R.Dead when r.dead -> ()
  | Ok a -> failf "claim answered %a" R.pp_answer a
  | Error e -> raise e

let export b = ignore (B.export b Bigarray.char)

let take_write b =
  match Rig_support.donated b with
  | None -> "refused"
  | Some b' ->
      B.write b' Bigarray.char (fun a -> Bigarray.Array1.fill a 'd');
      "ended"

let free b = if B.free b then "ended" else "kept"

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
    command "take" (memory ^-> judges string) judge_take take_write;
    command "free" (memory ^-> judges string) judge_take free;
  ]

(* Submissions *)

(* A take of each inside the submit's wait is refused, and one after it
   succeeds. *)
let test_submit_claims () =
  let producer, pp = P.open_ "claim:submit-claims-producer" in
  let d, _ = P.open_ "claim:submit-claims" in
  let buffers = Array.init 5 (fun _ -> B.create d 64) in
  let s = Sub.make d [| Rig_support.touching (Array.make 5 Sub.read) |] in
  let point = submit (Sub.make producer [||]) in
  let takes () = Array.to_list (Array.map alone buffers) in
  let during = ref None in
  P.interrupt pp;
  let before =
    Sys.signal Sys.sigint
      (Sys.Signal_handle (fun _ -> during := Some (takes ())))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      ignore (submit s ~buffers ~args:(Rig_support.zeros 1) ~waits:[| point |]));
  equal ~msg:"inside the submit's wait"
    (option (list string))
    (Some (List.init 5 (fun _ -> "refused")))
    !during;
  equal ~msg:"after the submit" (list string)
    (List.init 5 (fun _ -> "taken"))
    (takes ())

(* Hold submit at a foreign wait, after it has collected the source's uses. A
   take must not overwrite the source before that submit publishes its own use,
   whether the submission names the source as fixed memory or a copy part names
   it. *)
let test_submit_take (name, kind) =
  let producer, pp = P.open_ ("claim:submit-producer:" ^ name) in
  let d, p = P.open_ ("claim:submit:" ^ name) in
  let src = B.create d 8 and dst = B.create d 8 in
  Rig_support.store (R.host src) 42;
  Rig_support.store (R.host dst) 0;
  let s, buffers, args =
    match kind with
    | `Copy ->
        ( Sub.make d [| Sub.(copy ~src:(Slot 0) ~dst:(Slot 1)) |],
          [| src; dst |],
          [||] )
    | `Fixed ->
        let s =
          Sub.make
            ~fixed:[ (src, Rig.Read); (dst, Rig.Read_write) ]
            d
            [| Sub.fill Rig_support.carry ~params:24 |]
        in
        let params = Bytes.create 24 in
        Bytes.set_int64_le params 0 (Int64.of_int (Sub.address s dst));
        Bytes.set_int64_le params 8 (Int64.of_int (Sub.address s src));
        Bytes.set_int64_le params 16 8L;
        (s, [||], [| { Rig_support.zero with params } |])
  in
  (* A submission keeps its fixed memory for its life, and claims a submit's
     buffers only while the submit runs. *)
  if kind = `Fixed then
    equal ~msg:"the source once the submission is made" string "refused"
      (alone src);
  let point = submit (Sub.make producer [||]) in
  P.gate pp;
  let waiting =
    Domain.spawn (fun () -> Rig.submit s ~waits:[| point |] ~buffers ~args)
  in
  let took = ref "refused" in
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
          took := "taken";
          B.wait replacement Rig.Read_write;
          Rig_support.store (R.host replacement) 99);
  equal ~msg:"the take and the submitted copy's result" (pair string int)
    ("refused", 42)
    (!took, Rig_support.load (R.host dst));
  equal ~msg:"the submitted work completed" int 0 (P.queued p);
  equal ~msg:"the source after hand-over" string
    (if kind = `Copy then "taken" else "refused")
    (alone src)

(* Each buffer passed in two slots. *)
let test_run_claim_refusals () =
  let d, p = P.open_ "claim:run-refusals" in
  let buffers () = Array.init 5 (fun _ -> B.create d 8) in
  let s = Sub.make d [| Rig_support.touching (Array.make 10 Sub.read) |] in
  let slots bs = Array.init 10 (fun i -> bs.(i / 2)) in
  let check bs =
    equal ~msg:"takes after the submit" (list string)
      (List.init 5 (fun _ -> "taken"))
      (Array.to_list (Array.map alone bs))
  in
  let args = Rig_support.zeros 1 in
  let bs = buffers () in
  ignore (submit s ~buffers:(slots bs) ~args);
  check bs;
  for i = 0 to 4 do
    let bs = buffers () in
    let replacement = taken ~why:"replaced" bs.(i) in
    raises_match (dead "replaced") (fun () ->
        submit s ~buffers:(slots bs) ~args);
    bs.(i) <- replacement;
    check bs
  done;
  ignore (P.run p)

let test_run_wait_raises () =
  let producer, pp = P.open_ "claim:run-raises-producer" in
  let d, p = P.open_ "claim:run-raises" in
  let a = B.create d 8 and b = B.create d 8 in
  let s = Sub.make d [| Rig_support.touching (Array.make 3 Sub.read) |] in
  let args = Rig_support.zeros 1 in
  let point = submit (Sub.make producer [||]) in
  P.interrupt pp;
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      raises Exit (fun () ->
          Rig.submit s ~waits:[| point |] ~buffers:[| a; b; a |] ~args));
  let a' = Rig_support.donated a and b' = Rig_support.donated b in
  equal ~msg:"takes after it"
    (list (option string))
    [ Some "donated"; Some "donated" ]
    (deaths [ a; b ]);
  let a = require_some a' and b = require_some b' in
  ignore (P.run pp);
  ignore (Rig.submit s ~waits:[| point |] ~buffers:[| a; b; a |] ~args);
  ignore (P.run p)

(* None of them is taken while the submission is reachable, and each is once it
   is collected. Slots keep nothing. *)
let test_submission_keeps () =
  let d, _ = P.open_ "claim:submission-keeps" in
  let fixed = B.create d 8 and src = B.create d 8 and dst = B.create d 8 in
  let ws = B.create Rig.host 8 and slot = B.create d 8 in
  let s =
    Sub.(
      make
        ~fixed:[ (fixed, Rig.Read) ]
        d
        [|
          copy ~src:(Buffer src) ~dst:(Buffer dst);
          words ws;
          copy ~src:(Slot 0) ~dst:(Buffer dst);
        |])
  in
  equal ~msg:"while the submission is reachable" (list string)
    [ "refused"; "refused"; "refused"; "refused"; "taken" ]
    (List.map alone [ fixed; src; dst; ws; slot ]);
  ignore (Sys.opaque_identity (submit s ~buffers:[| B.create d 8 |]));
  collect_keeps ();
  equal ~msg:"once it is collected" (list string)
    [ "taken"; "taken"; "taken"; "taken" ]
    (List.map alone [ fixed; src; dst; ws ])

(* rig_buffer_claim *)

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

let claims_device = fixture (fun () -> P.open_ "claim:claim-all")
let read_only = fixture (fun () -> Rig_support.io "claim:claim-all-read-only")

(* A buffer in [s] and the access it is claimed for. *)
let buffer_in s =
  let d, _ = claims_device () in
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
      (Rig_support.io_buffer ~access:Rig.Read (read_only ()) 8, Rig.Read_write)

(* [rig_buffers_claim] refuses at the first buffer that [rig_buffer_claim] would
   refuse, holding nothing, and otherwise answers each buffer as
   [rig_buffer_claim] would and holds them all until [rig_buffers_release]. *)
let claim_all_law ss =
  let bs = List.map buffer_in ss in
  let buffers = Array.of_list (List.map fst bs) in
  let access = Array.of_list (List.map snd bs) in
  let expected =
    List.find_mapi (fun i s -> Option.map (fun a -> (i, a)) (refusal s)) ss
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
      equal ~msg:"takes under the claims" (list string)
        (List.map (fun _ -> "refused") bs)
        (List.map (fun (b, _) -> alone b) bs);
      R.release_all buffers
  | R.All _, Some _ -> fail "every buffer claimed past a refusal"
  | R.Refused (i, a), None ->
      failf "buffer %d refused (%a) with none to refuse" i R.pp_answer a);
  List.iter2
    (fun s (b, _) ->
      match s with
      | Live _ | Pending ->
          equal ~msg:"a take after the claims" string "taken" (alone b)
      | Dead | Read_only -> ())
    ss bs;
  ignore (P.run (snd (claims_device ())))

(* It sits beside other readers, refuses a take until it is released, and is
   refused on a dead buffer, where it leaves no claim behind. The buffer a take
   gives is claimed at once. *)
let test_c_claims () =
  let b = B.create Rig.host 16 in
  equal ~msg:"for reading" answer R.Claimed (R.claim b Rig.Read);
  equal ~msg:"for writing" answer R.Claimed (R.claim b Rig.Read_write);
  equal ~msg:"a take under two claims" string "refused" (take_of [ b ]);
  R.release b;
  R.release b;
  let b' = taken b in
  equal ~msg:"the taken buffer" answer R.Claimed (R.claim b' Rig.Read);
  R.release b';
  equal ~msg:"dead" answer R.Dead (R.claim b Rig.Read);
  equal ~msg:"its reason" (option string) (Some "donated") (R.why b);
  equal ~msg:"a take after it" string "taken" (alone b')

(* It answers Wait while a point the access must follow is unreached, the last
   write for reading and every use for writing, and holds the claim then;
   Claimed once the word shows the point. A wait under the claim returns once
   the points are reached. *)
let test_c_wait () =
  let d, p = P.open_ "claim:c-wait" in
  let m = B.create d 64 in
  write d m;
  equal ~msg:"an unreached write" answer R.Wait (R.claim m Rig.Read);
  equal ~msg:"a take while it waits" string "refused" (alone m);
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
  equal ~msg:"a take once released" string "taken" (alone m)

(* Work that ran answers Wait until a wait reads the word, and Claimed after
   it. *)
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

(* Even once its work on it was reached: the wait under it raises Lost. *)
let test_c_lost () =
  let d, p = P.open_ "claim:c-lost" in
  let m = B.create d 64 in
  write d m;
  ignore (P.run p);
  lose d p;
  equal ~msg:"reached, lost" answer R.Lost (R.claim m Rig.Read);
  raises_match (lost_by d) (fun () -> R.wait m Rig.Read);
  claim_release m

let test_c_lost_waiting () =
  let d, p = P.open_ "claim:c-lost-waiting" in
  let m = B.create d 64 in
  write d m;
  equal ~msg:"unreached" answer R.Wait (R.claim m Rig.Read);
  P.fault p "gone";
  raises_match lost (fun () -> R.wait m Rig.Read);
  claim_release m

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
      equal ~msg:"a take while the wait is blocked" string "refused"
        (take_of [ m ]));
  equal ~msg:"the write is done" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  equal ~msg:"a take under the claim" string "refused" (take_of [ m ]);
  claim_release m;
  equal ~msg:"a take once released" string "taken" (take_of [ m ])

(* Its stop's last value in the word reaches nothing. *)
let test_c_lost_points () =
  let d, p = P.open_ "claim:c-lost-points" in
  let reached = B.create Rig.host B.paged_from in
  let unreached = B.create Rig.host B.paged_from in
  write d (require_some (B.borrow d reached));
  ignore (P.run p);
  write d (require_some (B.borrow d unreached));
  lose d p;
  equal ~msg:"reached" answer R.Claimed (R.claim reached Rig.Read_write);
  R.release reached;
  equal ~msg:"unreached" answer R.Lost (R.claim unreached Rig.Read);
  raises_match (lost_by d) (fun () -> R.wait unreached Rig.Read);
  R.release unreached

let test_c_lost_reads () =
  let d, p = P.open_ "claim:c-lost-reads" in
  let m = B.create Rig.host B.paged_from in
  read d (require_some (B.borrow d m));
  lose d p;
  equal ~msg:"for reading" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  equal ~msg:"for writing" answer R.Lost (R.claim m Rig.Read_write);
  R.release m

(* Whichever device's use came first, and the wait raises the lost device's Lost
   at once: the live device's sleeps block at its gate, so a wait for its work
   would never return. *)
let test_c_lost_behind_live () =
  let order lost_first =
    let name = if lost_first then "claim:c-behind-1" else "claim:c-behind-2" in
    let live, lp = P.open_ (name ^ "-live") in
    let d, p = P.open_ (name ^ "-lost") in
    let m = B.create Rig.host B.paged_from in
    let read_on d = read d (require_some (B.borrow d m)) in
    if lost_first then (
      read_on d;
      read_on live)
    else (
      read_on live;
      read_on d);
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

let test_c_lost_borrow () =
  let d, p = P.open_ "claim:c-lost-borrow" in
  let m = B.create Rig.host B.paged_from in
  let b = require_some (B.borrow d m) in
  lose d p;
  equal ~msg:"the borrow" answer R.Lost (R.claim b Rig.Read);
  raises_match (lost_by d) (fun () -> R.wait b Rig.Read);
  R.release b;
  equal ~msg:"the memory" answer R.Claimed (R.claim m Rig.Read_write);
  R.release m

(* A claim to read waits for an unreached submission that writes it, and for
   none that reads it. *)
let test_c_fixed () =
  let d, p = P.open_ "claim:c-fixed" in
  let m = B.create d 64 in
  let fixed access = Sub.make ~fixed:[ (m, access) ] d [||] in
  ignore (submit (fixed Rig.Read));
  equal ~msg:"an unreached read" answer R.Claimed (R.claim m Rig.Read);
  R.release m;
  ignore (submit (fixed Rig.Read_write));
  equal ~msg:"an unreached write" answer R.Wait (R.claim m Rig.Read);
  R.release m;
  ignore (P.run p);
  equal ~msg:"reached" answer R.Claimed (R.claim m Rig.Read);
  R.release m

(* Also once a later device holds its slot: the C claim reads the slot as the
   device's only while the slot's line names it. The later device's first value
   shows that it took the slot. *)
let test_closed_slot_claim () =
  let d, _ = P.open_ "claim:closed" in
  let p = Rig.submit (Sub.make d [||]) ~buffers:[||] ~args:[||] in
  Rig.Point.wait p;
  let b = B.create d 0 in
  Rig.close d;
  Rig.free_cache d;
  let later, _ = P.open_ "claim:later" in
  equal ~msg:"the later device took the closed one's slot" int
    (Rig.Point.value p) (Rig.submitted later);
  equal ~msg:"the C claim" answer R.Lost (R.claim b Rig.Read);
  Rig.close later

let tests =
  [
    group "read claims"
      [
        cases ~id:fst "a take inside an access's function is refused"
          in_function test_take_in_function;
        cases ~id:fst
          "a host access waiting for work holds its memory from a take"
          host_accesses test_host_access_waiting;
        test "a host copy waiting holds its source from a take"
          test_host_copy_source;
        test "a host copy waiting holds its destination from a take"
          test_host_copy_destination;
        stateful ~domains:2
          "host copies, blits and fills beside takes are ordered with them"
          copy_commands;
        test "work on a device holds no claim, and a taker's write waits for it"
          test_device_work;
        test "a read whose wait raises leaves no claim" test_wait_raises;
        test "a signal pending as a read's function returns leaves no claim"
          test_signal_returned;
        test "a signal pending as a read's function raises leaves no claim"
          test_signal_raised;
        test
          "an exception at any allocation of a blit into Mapped memory leaves \
           no claim"
          test_walk_blit;
        test
          "an exception at any allocation of a fill of Mapped memory leaves no \
           claim"
          test_walk_fill;
      ];
    group "keep"
      [
        test "a keep holds memory from takes until it is collected" test_keep;
        test "a keep of a dead buffer raises and keeps none" test_keep_refusals;
        test "a release ends a keep before its collection" test_release;
        test "a later release and the collection of a released keep end nothing"
          test_release_once;
        test
          "releases of one keep on two domains beside its collection end it \
           once"
          test_releases_beside_drain;
        stateful ~count:70 ~steps:1 ~domains:2
          "releases of one keep on two domains return once its holds have ended"
          release_commands;
        stateful ~domains:2
          "keeps beside takes of one memory are ordered with them" keep_commands;
      ];
    group "outside the claims"
      [
        test "a bigarray's memory is never taken" test_of_bigarray;
        test "memory shared through any of its buffers is never taken"
          test_share_covers;
        test "memory dma described or refused is never taken" test_dma;
        test "a share of a dead buffer raises" test_share_refused;
      ];
    group "take"
      [
        prop ~count:300 ~examples:after_held
          "a take takes the memory of every buffer or of none" case take_law;
        test "a take of memory that admits only reads is refused" test_read_only;
        test "a take naming one memory twice is refused" test_take_overlap;
        test "memories with no address are told apart" test_no_address;
        test "a taker uses the memory through the buffer the take gives"
          test_taken_writes;
        test "a borrow and the memory it maps share one count"
          test_borrow_counts;
        test "a borrow of a taken buffer raises" test_borrow_taken;
        test "a take of memories of two devices takes both or neither"
          test_shards;
        test "an export of taken memory is the taker's" test_export_taken;
        test "a taker submits work on the buffer the take gives"
          test_taker_submits;
        cases "the buffer a take gives is used as any other" ~id:taken_name
          taken_cases taken_law;
        test "memory taken twice lives while its last buffer does"
          test_take_twice;
        test "a dead buffer refuses claims and keeps" test_dead_claims;
        test "a take of memory a loss reaches raises and takes nothing"
          test_lost_take;
        test "an exception at any allocation of a take takes all or none"
          test_walk_take;
        stateful ~domains:2
          "takes of two memories in either order beside reads are ordered with \
           them"
          group_commands;
        stateful ~domains:2
          "reads, exports and shares beside a take or a free are ordered with \
           it"
          take_commands;
      ];
    group "submissions"
      [
        test "a submit holds a read claim on its buffers until it returns"
          test_submit_claims;
        cases ~id:fst "a submit's memory refuses a take while the submit runs"
          [ ("fixed", `Fixed); ("copied", `Copy) ]
          test_submit_take;
        test "a submit of a dead buffer raises and releases every claim it took"
          test_run_claim_refusals;
        test "a submit whose wait raises releases its buffers' claims"
          test_run_wait_raises;
        test "a submission keeps its fixed memory and its parts' buffers"
          test_submission_keeps;
      ];
    group "rig_buffer_claim"
      [
        prop ~count:300 "rig_buffers_claim claims every buffer or none" states
          claim_all_law;
        test "a claim from C is a read claim" test_c_claims;
        test
          "a claim from C holds the memory and waits until the word shows its \
           points"
          test_c_wait;
        test "behind a transport a claim reads the word a wait read"
          test_c_transport;
        test "a claim on a lost device's memory answers Lost under the claim"
          test_c_lost;
        test "a device lost during a wait from C raises Lost under the claim"
          test_c_lost_waiting;
        test "a wait from C holds its claim while it waits" test_wait;
        test "a claim on memory a lost device did not reach answers Lost"
          test_c_lost_points;
        test "a lost device's unreached read is lost for writing only"
          test_c_lost_reads;
        test "a loss behind a live device's unreached work answers Lost"
          test_c_lost_behind_live;
        test "a borrow on a lost device answers Lost" test_c_lost_borrow;
        test "a claim on fixed memory follows its uses by their access"
          test_c_fixed;
        test "the C claim of a closed device's buffer answers its loss"
          test_closed_slot_claim;
      ];
  ]

let () = exit (run "rig.claim" tests)
