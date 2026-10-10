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

let timeout = 60.
let why = "the walk's failure"
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

(* How a walk fails an operation: from its [n]-th fallible call on, every call
   faults; its [n]-th call refuses; from its [n]-th call on, every call refuses,
   as a device out of memory does; or a function it gave rig raises [Exit]. *)
type failure = Fault of int | Refuse of int | Exhaust of int | Raise

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

(* An operation on a device: [prepare d] makes what it needs, and is the
   operation, told the failure it meets. It checks the values it gets, and
   raises on one rig.mli does not allow. [raises] says whether it gives rig a
   function, which raises under [Raise]. *)
type op = {
  name : string;
  opener : string -> Rig.t * P.t;
  prepare : Rig.t -> failure option -> unit;
  raises : bool;
}

let fresh =
  let k = ref 0 in
  fun what ->
    incr k;
    Printf.sprintf "walk:%s:%d" what !k

let page = 1 lsl 16
let pattern n = String.init n (fun i -> Char.chr (i land 0xff))

let collect () =
  Gc.full_major ();
  Gc.full_major ()

let bytes b =
  let ba = B.bigarray Bigarray.char b in
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let filled n =
  let b = B.create Rig.host n in
  String.iteri (Bigarray.Array1.set (B.bigarray Bigarray.char b)) (pattern n);
  b

let op ?(opener = fun name -> P.open_ name) ?(raises = false) name prepare =
  { name; opener; prepare; raises }

(* A refusal may reach a call that answers [None] or [Error]. *)
let refusing = function Some (Refuse _ | Exhaust _) -> true | _ -> false

let create memory =
  op
    (match memory with
    | B.Device -> "create"
    | Pinned -> "create pinned"
    | Mapped -> "create mapped")
    (fun d _ -> equal int 64 (B.length (B.create ~memory d 64)))

let borrow_of name make =
  op name (fun d ->
      let b = make () in
      fun failure ->
        match B.borrow d b with
        | Some b' -> equal int (B.length b) (B.length b')
        | None -> equal ~msg:"borrow refused" bool true (refusing failure))

(* The device whose memory [borrow peer] borrows, one for every walk. *)
let peer = lazy (fst (P.open_ "walk:peer"))
let borrow_host = borrow_of "borrow host" (fun () -> B.create Rig.host page)

let borrow_peer =
  borrow_of "borrow peer" (fun () -> B.create (Lazy.force peer) 64)

let bump arg =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work = Fill { fill = S.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let once ~run s = Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||]

let submit =
  op "submit" (fun d ->
      let arg = B.create Rig.host 8 in
      Bigarray.Array1.fill (B.bigarray Bigarray.char arg) '\000';
      let s = Sub.make ~reads:0 ~writes:0 d [| bump arg |] in
      let run = Sub.Run.make () in
      fun _ ->
        Rig.Point.wait (once ~run s);
        equal string "\001\000\000\000\000\000\000\000" (bytes arg))

(* A device whose memory the host does not address, which copied through the
   staging memory both ways once, its mappings of host memory refused: what it
   made for those copies lasts as long as the staging memory, for good. *)
let staging name =
  let d, p = P.open_ ~host_visible:false name in
  let on_d = B.create d page in
  P.fail_at p 1 (`Refuse 1);
  B.copy ~src:(B.create Rig.host page) ~dst:on_d;
  P.fail_at p 1 (`Refuse 1);
  B.copy ~src:on_d ~dst:(B.create Rig.host page);
  (d, p)

(* Copies between the host and [d], whose memory the host does not address: [d]
   copies, through its mapping of the host's memory or through the staging
   memory. *)
let copy direction =
  let name =
    match direction with `Into -> "copy into" | `Out_of -> "copy out of"
  in
  op name ~opener:staging (fun d ->
      let on_d = B.create d page in
      let src, dst =
        match direction with
        | `Into -> (filled page, on_d)
        | `Out_of ->
            B.copy ~src:(filled page) ~dst:on_d;
            (on_d, B.create Rig.host page)
      in
      fun _ ->
        B.copy ~src ~dst;
        let back = B.create Rig.host page in
        B.copy ~src:dst ~dst:back;
        equal string (pattern page) (bytes back))

let load =
  op "load" (fun d failure ->
      match Rig.Image.load d "code:64" with
      | Ok p -> (
          match Rig.Image.entry p "main" with
          | Some _ -> ()
          | None -> equal ~msg:"entry refused" bool true (refusing failure))
      | Error _ -> equal ~msg:"load refused" bool true (refusing failure))

let wait_transport =
  op "wait behind a transport" ~opener:(P.open_ ~transport:true) (fun d ->
      let s = Sub.make ~reads:0 ~writes:0 d [||] in
      let run = Sub.Run.make () in
      fun _ -> Rig.Point.wait (once ~run s))

(* Submits on [d] once with a hold whose release raises under [Raise],
   leaving the hold and its submission unreachable. *)
let[@inline never] submit_held d failure =
  let release () = if failure = Some Raise then raise Exit in
  let h = Rig.Hold.make ~release () in
  let run = Sub.Run.make () in
  Rig.Point.wait (once ~run (Sub.make ~hold:h ~reads:0 ~writes:0 d [||]))

let hold =
  op "hold" ~raises:true (fun d failure ->
        submit_held d failure;
        collect ();
        ignore (B.create d 0))

(* A submission with a part under a profile: its hand-over takes a time
   pair, and the profile reads it once the value is reached. *)
let profile =
  op "profile" (fun d ->
      let arg = B.create Rig.host 8 in
      let s = Sub.make ~reads:0 ~writes:0 d [| bump arg |] in
      let run = Sub.Run.make () in
      fun _ ->
        ignore (Rig.Profile.take (fun () -> Rig.Point.wait (once ~run s))))

let ops =
  [
    create B.Device;
    create B.Pinned;
    create B.Mapped;
    borrow_host;
    borrow_peer;
    submit;
    load;
    wait_transport;
    hold;
    profile;
  ]

(* Walking *)

let census () =
  collect ();
  (S.heap_bytes (), S.descriptors ())

let count call p = List.length (List.filter (( = ) call) (P.log p))

(* The regions [p] still holds of the sizes the operations use: the staging
   memory, which the host keeps for good, has its own. *)
let left p = List.filter (fun n -> n = 64 || n = page) (P.outstanding p)

(* Collects what the caller dropped, then returns memory in the order it comes
   back: a drain of the host (and with it of the lost devices) and of the peer
   moves [d]'s mappings of their memory into [d]'s cache, and [d]'s cache
   returns them; the memory they mapped is then unreachable, and the host's next
   drain takes it back. *)
let settle d =
  collect ();
  let drain d = if Rig.lost d = None then ignore (B.create d 0) in
  let devices =
    (if Lazy.is_val peer then [ Lazy.force peer ] else []) @ [ d ]
  in
  drain Rig.host;
  List.iter drain devices;
  List.iter Rig.free_cache devices;
  collect ();
  drain Rig.host;
  Rig.free_cache Rig.host

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
  let make () = if failure = Raise then raise Exit else Ok p in
  let got =
    match Rig.open_ (module P) ~name make with
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
  ignore (require_ok (Rig.open_ (module P) ~name (fun () -> Ok (P.make ()))));
  match (heap, heap') with Some h, Some h' -> h' - h | _ -> 0

(* A fact refuses nothing: the walk faults them, and raises in the opener. *)
let test_open () =
  ignore (open_attempt (Fault 1));
  let p = P.make () in
  ignore
    (require_ok (Rig.open_ (module P) ~name:(fresh "open") (fun () -> Ok p)));
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
  let d = require_ok (Rig.memory_device name) in
  let counter = B.create Rig.host 8 in
  S.store (B.address counter) 0;
  let fill =
    Sub.Fill
      { fill = S.countdown; arg = counter; ring_units = 0; segment_bytes = 0 }
  in
  let part = { Sub.queue = "COMPUTE:0"; after = [||]; work = fill } in
  let s = ref (Some (Sub.make ~reads:0 ~writes:0 d (Array.make fills part))) in
  let run = Sub.Run.make () in
  Rig.Point.wait (once ~run (Option.get !s));
  let heap, fds = census () in
  S.store (B.address counter) n;
  let got =
    outcome d (fun () -> Rig.Point.wait (once ~run (Option.get !s)))
  in
  let at = Printf.sprintf ", fill %d" n in
  equal ~msg:("outcome" ^ at) string "lost: a fill failed" got;
  settle d;
  let heap', fds' = census () in
  equal ~msg:("descriptors" ^ at) (option int) fds fds';
  s := None;
  let d = require_ok (Rig.memory_device name) in
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
  let d = require_ok (Rig.memory_device (fresh "memory")) in
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

(* A copy through the staging memory needs [d] to map it. *)
let staged = [ copy `Into; copy `Out_of ]

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
