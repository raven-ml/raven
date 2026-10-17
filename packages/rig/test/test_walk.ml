(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Failure walks. An operation runs once for each fallible call it makes through
   Polled, with that call failing ({!Rig_support.Polled.fail_at}), four times
   each; an operation that gives rig a function runs once more with that
   function raising. After each failure the outcome is the one rig.mli states,
   the driver holds no region of the operation's, a lost device was stopped
   once, the C heap and the descriptors are back where they were, and the
   operation runs again. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module S = Rig_support
open Ops

let timeout = 60.
let reps = 4

(* A failure may leave the C heap grown by what a lost device keeps for good,
   its reason (32 bytes for [why]), and by the allocator's rounding. A leak
   shows in every run of a failure, while the allocator's own caches (glibc's
   per-thread ones) move single runs either way: the walk checks the least
   growth of [reps] runs. *)
let heap_slack = 128

(* [grew what deltas] checks the C heap's growth over runs of a failure,
   [deltas] the growth of each. *)
let grew what deltas =
  let each = String.concat ", " (List.map string_of_int deltas) in
  less
    ~msg:(Printf.sprintf "C heap growth, %s (%s)" what each)
    int ~than:heap_slack
    (List.fold_left Int.min max_int deltas)

(* Failures *)

let pp_failure ppf = function
  | Fault n -> Format.fprintf ppf "a fault at call %d" n
  | Refuse n -> Format.fprintf ppf "a refusal at call %d" n
  | Exhaust n -> Format.fprintf ppf "refusals from call %d" n
  | Raise -> Format.fprintf ppf "a raising function"

let inject p = function
  | Fault n -> P.fail_at p n (`Fault why)
  | Refuse n -> P.fail_at p n (`Refuse 1)
  | Exhaust n -> P.fail_at p n (`Refuse max_int)
  | Raise -> ()

(* The outcomes rig.mli allows for each failure. *)
let allowed = function
  | Fault _ -> [ "lost: " ^ why ]
  | Refuse _ -> [ "done" ]
  | Exhaust _ -> [ "done"; "out of memory" ]
  | Raise -> [ "raised Exit" ]

let outcome d f =
  match f () with
  | () -> "done"
  | exception Rig.Lost (d', w) when Rig.equal d d' -> "lost: " ^ w
  | exception Rig.Out_of_memory (d', _) when Rig.equal d d' -> "out of memory"
  | exception Exit -> "raised Exit"

(* Operations *)

(* Walking *)

let census () =
  collect ();
  (S.heap_bytes (), S.descriptors ())

(* The regions [p] still holds of the sizes the operations use: the staging
   memory, which the host keeps for good, has its own. *)
let left p = List.filter (fun n -> n = 64 || n = page) (P.outstanding p)

(* Runs [op] on a device it ran on once, which made what the device keeps for
   good, with [failure]; checks what the failure left, and runs [op] again: the
   growth of the C heap. *)
let attempt op failure =
  let name = fresh op.name in
  let d, p = op.opener name in
  op.prepare d None;
  settle d;
  let heap, fds = census () in
  let run = ref (Some (op.prepare d)) in
  inject p failure;
  let got = outcome d (fun () -> Option.get !run (Some failure)) in
  run := None;
  let at = Format.asprintf ", %a" pp_failure failure in
  mem ~msg:("outcome" ^ at) string got (allowed failure);
  let lost = match failure with Fault _ -> Some why | _ -> None in
  equal ~msg:("lost" ^ at) (option string) lost (Rig.lost d);
  (* A lost device's stop runs at its next call of its own, its close here. *)
  if lost <> None then Rig.close d;
  settle d;
  equal ~msg:("regions left with the driver" ^ at) (list int) [] (left p);
  equal ~msg:("stops" ^ at) int (if lost = None then 0 else 1) (count "stop" p);
  let heap', fds' = census () in
  equal ~msg:("descriptors" ^ at) (option int) fds fds';
  P.fail_at p 1 (`Refuse 0);
  let d = if lost = None then d else fst (op.opener name) in
  op.prepare d None;
  settle d;
  match (heap, heap') with Some h, Some h' -> h' - h | _ -> 0

(* The fallible calls [op] makes on a device it ran on once. *)
let steps op =
  let d, p = op.opener (fresh op.name) in
  op.prepare d None;
  settle d;
  let run = op.prepare d in
  let before = P.steps p in
  run None;
  let k = P.steps p - before in
  settle d;
  k

let all = [ (fun n -> Fault n); (fun n -> Refuse n); (fun n -> Exhaust n) ]

(* Walks [op] through the failures [kinds] make of each of its fallible calls,
   and through a raising function if it gives rig one. *)
let walk ?(kinds = all) op () =
  (* A first loss and a first refusal make what the process keeps for good. *)
  ignore (attempt op (Fault 1));
  ignore (attempt op (Refuse 1));
  let k = steps op in
  greater ~msg:"fallible calls" int ~than:0 k;
  let failures n = List.map (fun kind -> kind n) kinds in
  let failures =
    List.concat_map failures (List.init k succ)
    @ if op.raises then [ Raise ] else []
  in
  List.iter
    (fun failure ->
      grew
        (Format.asprintf "%a" pp_failure failure)
        (List.init reps (fun _ -> attempt op failure)))
    failures

(* Opening: a fact the open reads fails, or the function that opens the driver
   raises. A failed open leaves no device and the C heap as it was, stops once a
   driver whose fact failed, and leaves the name to open. *)
let open_attempt failure =
  let name = fresh "open" in
  let p = P.make () in
  let heap, fds = census () in
  inject p failure;
  let make ~word =
    if failure = Raise then raise Exit else Ok (P.bind p ~word)
  in
  let got =
    match Rig.open_ { name; driver = (module P); make } with
    | Ok _ -> "done"
    | Error e -> "error: " ^ e
    | exception Exit -> "raised Exit"
  in
  let at = Format.asprintf ", %a" pp_failure failure in
  let expected =
    match failure with
    | Raise -> "raised Exit"
    | _ -> Printf.sprintf "error: %s: %s" name why
  in
  equal ~msg:("outcome" ^ at) string expected got;
  let stops = match failure with Fault _ -> [ "stop" ] | _ -> [] in
  equal ~msg:("driver calls" ^ at) (list string) stops (P.log p);
  let heap', fds' = census () in
  equal ~msg:("descriptors" ^ at) (option int) fds fds';
  ignore
    (require_ok
       (Rig.open_
          (P.opener name (P.make ()))));
  match (heap, heap') with Some h, Some h' -> h' - h | _ -> 0

(* A fact refuses nothing: the walk faults them, and raises in the opener. *)
let test_open () =
  ignore (open_attempt (Fault 1));
  let p = P.make () in
  ignore
    (require_ok
       (Rig.open_
          (P.opener (fresh "open") p)));
  let k = P.steps p in
  greater ~msg:"facts" int ~than:0 k;
  List.iter
    (fun failure ->
      grew
        (Format.asprintf "%a" pp_failure failure)
        (List.init reps (fun _ -> open_attempt failure)))
    (Raise :: List.init k (fun i -> Fault (i + 1)))

(* A memory device runs a submission's fills in its hand-over: the [n]-th of
   three failing loses it, leaves the C heap as it was, and its name opens
   again. *)
let fills = 3

let memory_attempt n =
  let name = fresh "memory" in
  let d = Result.get_ok (Rig.memory_device name) in
  let counter = B.create d 8 in
  S.store (S.Reader.host counter) 0;
  let fill =
    Sub.fill ~refs:[| Sub.write 0 ~at:0 |] S.countdown ~params:8
  in
  let part = fill in
  let s =
    ref (Some (Sub.make d (Array.make fills part)))
  in
  let run = Sub.Run.make () and buffers = [| counter |] in
  for i = 0 to fills - 1 do
    Sub.Run.int64 run (Sub.block (Option.get !s) i) 0 0
  done;
  Rig.Point.wait (once ~buffers ~run (Option.get !s));
  let heap, fds = census () in
  S.store (S.Reader.host counter) n;
  let got =
    outcome d (fun () -> Rig.Point.wait (once ~buffers ~run (Option.get !s)))
  in
  let at = Printf.sprintf ", fill %d" n in
  equal ~msg:("outcome" ^ at) string "lost: a fill failed" got;
  settle d;
  let heap', fds' = census () in
  equal ~msg:("descriptors" ^ at) (option int) fds fds';
  s := None;
  let d = Result.get_ok (Rig.memory_device name) in
  equal ~msg:("open again" ^ at) (option string) None (Rig.lost d);
  match (heap, heap') with Some h, Some h' -> h' - h | _ -> 0

let test_memory_fill () =
  ignore (memory_attempt 1);
  for n = 1 to fills do
    grew
      (Printf.sprintf "fill %d" n)
      (List.init reps (fun _ -> memory_attempt n))
  done

(* A memory device whose allocation the host refuses raises Out_of_memory, and
   allocates the next buffer that fits. *)
let test_memory_refused () =
  let d = Result.get_ok (Rig.memory_device (fresh "memory")) in
  ignore (B.create d 64);
  settle d;
  let heap, fds = census () in
  raises_match
    (function Rig.Out_of_memory (d', _) -> Rig.equal d d' | _ -> false)
    (fun () -> B.create d (1 lsl 60));
  settle d;
  let heap', fds' = census () in
  equal ~msg:"descriptors" (option int) fds fds';
  (match (heap, heap') with
  | Some h, Some h' -> grew "a refused allocation" [ h' - h ]
  | _ -> ());
  equal int 64 (B.length (B.create d 64))

(* On a device that never mapped the staging memory, every mapping refused. *)
let exhausted op =
  let op = { op with opener = P.open_ ~host_visible:false } in
  test
    (op.name ^ ", every mapping refused")
    (walk ~kinds:[ (fun n -> Exhaust n) ] op)

let () =
  exit
  @@ run "rig.walk"
       [
         group ~timeout "every failing call ends as rig.mli states"
           (List.map (fun op -> test op.name (walk op)) ops
           @ List.map
               (fun op ->
                 test op.name
                   (walk ~kinds:[ (fun n -> Fault n); (fun n -> Refuse n) ] op))
               staged
           @ List.map exhausted staged);
         group ~timeout "opening"
           [
             test "every failing fact or opener leaves the name to open"
               test_open;
           ];
         group ~timeout "memory devices"
           [
             test "every failing fill loses the device, which opens again"
               test_memory_fill;
             test "an allocation the host refuses raises Out_of_memory"
               test_memory_refused;
           ];
       ]
