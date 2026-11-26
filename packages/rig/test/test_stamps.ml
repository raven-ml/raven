(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Order and words, on Polled devices, whose queues run only when a wait sleeps
   or the test runs them, so a wait that returns early leaves its work unrun. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1
let names = Atomic.make 0

let fresh what =
  Printf.sprintf "stamps:%s-%d" what (Atomic.fetch_and_add names 1)

let software name = Result.get_ok (Rig.software (fresh name))

(* [b] borrowed on [d], as a submit on [d] takes it. *)
let on d b = require_some (B.borrow d b)

(* Submits on [d] over [buffers] one fill that does nothing and uses a slot per
   element of [uses] ({!Support.touching}). *)
let touch ?waits uses d buffers =
  submit ?waits (Sub.make d [| Support.touching uses |]) ~buffers ~args:one

(* Host buffers of [Rig.Buffer.paged_from] bytes start on a page, so devices
   borrow them. *)
let page_bytes = B.paged_from

(* The 64-bit word at [b]'s first byte, read by the host with no call of
   rig's. *)
let word b = Support.load (Support.Reader.host b)

(* A buffer's bytes as runs of one byte: [(c, n)] for [n] bytes [c]. *)
let runs_of b =
  B.read b Bigarray.char (fun a ->
      let runs = ref [] in
      for i = Bigarray.Array1.dim a - 1 downto 0 do
        match !runs with
        | (c, k) :: rest when c = a.{i} -> runs := (c, k + 1) :: rest
        | l -> runs := (a.{i}, 1) :: l
      done;
      !runs)

let runs = list (pair char int)

(* Device order *)

(* A value's work starts once the previous value's completed, on every queue: a
   copy on COPY:0, then a fill on COMPUTE:0 that reads what the copy wrote with
   unordered slots, sees the copy, and the other way round. Only the device's
   order orders them. *)
let test_device_order () =
  let d, p = P.open_ (fresh "device-order") in
  let n = page_bytes in
  let filled c =
    let b = B.create Rig.host n in
    B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a c);
    b
  in
  let a = filled 'a' and b = B.create d n in
  let c = filled '-' and out = filled '-' and z = filled 'z' in
  let fill dst src =
    let part =
      Sub.fill
        ~refs:[| Sub.unordered 0 ~at:0; Sub.unordered 1 ~at:8 |]
        Support.carry ~params:24
    in
    let params = Bytes.make 24 '\000' in
    Bytes.set_int64_le params 16 (Int64.of_int n);
    let args = [| { Support.zero with params } |] in
    ignore
      (submit ~args ~buffers:[| on d dst; on d src |] (Sub.make d [| part |]))
  in
  let copy src dst = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  let run parts = ignore (submit (Sub.make d parts)) in
  run [| copy (on d a) b |];
  fill out b;
  ignore (P.run p);
  equal ~msg:"a fill after a copy" runs [ ('a', n) ] (runs_of out);
  fill b z;
  run [| copy b (on d c) |];
  ignore (P.run p);
  equal ~msg:"a copy after a fill" runs [ ('z', n) ] (runs_of c)

(* Parts *)

(* A case of the order law: parts on a Polled device, copies whose ends are
   views of three allocations or slots, fills and launches whose refs name
   slots, words, and fixed memory. The allocations are two of the device's,
   which the host addresses only with [host_visible], and one of the host's,
   borrowed. A slot's refs are all of its kind, and a copy names only slots of
   ordered refs. A launch may also hold an entry ref. Each words part reads a
   host buffer of its own. *)
type view = { alloc : int; first : int; length : int }
type use = Read | Write | Unordered
type side = View of view | Slot of int

type order_part =
  | Copy of side * side
  | Fill of int list
  | Launch of int list * bool  (** Its slots, and whether an entry ref. *)
  | Words

type order_case = {
  host_visible : bool;
  kinds : use array;  (** Each slot's refs' kind. *)
  parts : order_part list;
  fixed : (view * Rig.access) list;
}

let pp_view ppf v = Format.fprintf ppf "%d[%d+%d]" v.alloc v.first v.length

let pp_order_case ppf c =
  let kind = function Read -> "r" | Write -> "w" | Unordered -> "u" in
  let side ppf = function
    | View v -> pp_view ppf v
    | Slot k -> Format.fprintf ppf "slot %d" k
  in
  let slots ppf ks =
    Format.pp_print_string ppf
      (String.concat ";"
         (List.map (fun k -> string_of_int k ^ kind c.kinds.(k)) ks))
  in
  let part ppf = function
    | Copy (s, d) -> Format.fprintf ppf "copy %a->%a" side s side d
    | Words -> Format.fprintf ppf "words"
    | Fill ks -> Format.fprintf ppf "fill [%a]" slots ks
    | Launch (ks, entry) ->
        Format.fprintf ppf "launch [%a]%s" slots ks
          (if entry then " entry" else "")
  in
  Format.fprintf ppf "%sparts [%a]; fixed [%a]"
    (if c.host_visible then "" else "unaddressed; ")
    (Format.pp_print_list ~pp_sep:Format.pp_print_space part)
    c.parts
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (v, a) ->
         Format.fprintf ppf "%a%s" pp_view v (if a = Rig.Read then "r" else "w")))
    c.fixed

(* Whether two views share a byte. *)
let share_byte v v' =
  v.alloc = v'.alloc && v.length > 0 && v'.length > 0
  && v.first < v'.first + v'.length
  && v'.first < v.first + v.length

(* Whether two views share memory, as rig orders it: a word, the [Rig.word]
   bytes at a multiple of it from the allocation's first byte. *)
let overlap v v' =
  let lo v = v.first / Rig.word * Rig.word
  and hi v = (v.first + v.length + Rig.word - 1) / Rig.word * Rig.word in
  v.alloc = v'.alloc && v.length > 0 && v'.length > 0
  && lo v < hi v'
  && lo v' < hi v

let nallocs = 3
let alloc_bytes = 64

let gen_order_case =
  let open Gen in
  (* Views start and end on 2-byte steps, so that views often share or abut
     their bounds, or share a word and no byte; empty views overlap nothing. *)
  let step = 2 in
  let view length =
    let+ alloc = int_range 0 (nallocs - 1)
    and+ first =
      map (( * ) step) (int_range 0 ((alloc_bytes - length) / step))
    in
    { alloc; first; length }
  in
  let length = map (( * ) step) (int_range 0 (alloc_bytes / step)) in
  (* Fills of unordered slots and read-only fixed memory leave copies free of
     them, so that a copy on the copy engine can follow one on a fill's
     queue. *)
  let* kinds =
    array ~size:(constant 3)
      (frequency
         [ (1, constant Read); (1, constant Write); (3, constant Unordered) ])
  in
  (* A slot of ordered refs for a copy's end, or none: slots run 8 bytes, so a
     copy with a slot end copies 8. *)
  let ordered = List.filter (fun k -> kinds.(k) <> Unordered) [ 0; 1; 2 ] in
  let slot_end =
    if ordered = [] then constant None
    else
      frequency
        [
          (3, constant None);
          (1, map Option.some (of_list ~pp:Format.pp_print_int ordered));
        ]
  in
  let slots =
    map
      (List.sort_uniq Int.compare)
      (list ~size:(int_range 0 2) (int_range 0 2))
  in
  let part =
    one_of
      [
        (* A copy's buffers share no byte and its slots differ
           (Submission.copy): a [dst] view that would share bytes with [src]
           moves to the next allocation, and a [dst] slot equal to [src]'s is a
           view. *)
        (let* src_slot, dst_slot = pair slot_end slot_end in
         let* length =
           if src_slot = None && dst_slot = None then length else constant 8
         in
         let+ src = view length and+ dst = view length in
         match (src_slot, dst_slot) with
         | Some k, Some k' when k <> k' -> Copy (Slot k, Slot k')
         | Some k, _ -> Copy (Slot k, View dst)
         | None, Some k -> Copy (View src, Slot k)
         | None, None ->
             let dst =
               if share_byte src dst then
                 { dst with alloc = (dst.alloc + 1) mod nallocs }
               else dst
             in
             Copy (View src, View dst));
        map (fun ks -> Fill ks) slots;
        (let+ ks = slots and+ entry = bool in
         Launch (ks, entry));
        constant Words;
      ]
  in
  let fixed =
    let* length = length in
    let+ v = view length
    and+ a =
      frequency [ (3, constant Rig.Read); (1, constant Rig.Read_write) ]
    in
    (v, a)
  in
  let+ host_visible = bool
  and+ parts = list ~size:(int_range 1 40) part
  and+ fixed =
    list
      ~size:(frequency [ (1, constant 0); (2, constant 1); (1, constant 3) ])
      fixed
  in
  { host_visible; kinds; parts; fixed }

(* The case's slots, renumbered from 0 in their order of use, so that every slot
   is a ref's or a copy's. *)
let slot_order c =
  let add acc k = if List.mem k acc then acc else acc @ [ k ] in
  let side acc = function Slot k -> add acc k | View _ -> acc in
  List.fold_left
    (fun acc -> function
      | Fill slots | Launch (slots, _) -> List.fold_left add acc slots
      | Copy (src, dst) -> side (side acc src) dst
      | Words -> acc)
    [] c.parts

(* What a part uses, as Submission.make states it: [`View] a buffer it names,
   [`Words] the host buffer a words part reads, which no other part names, and
   [`Slot] a submit's buffer, which may be any, with whether it writes. *)
let uses c p =
  let fixed = List.map (fun (v, a) -> (`View v, a = Rig.Read_write)) c.fixed in
  let side w = function View v -> (`View v, w) | Slot _ -> (`Slot, w) in
  match p with
  | Copy (src, dst) -> side false src :: side true dst :: fixed
  | Words -> (`Words, false) :: fixed
  | Fill slots | Launch (slots, _) ->
      List.filter_map
        (fun k ->
          match c.kinds.(k) with
          | Unordered -> None
          | Read -> Some (`Slot, false)
          | Write -> Some (`Slot, true))
        slots
      @ fixed

let conflict c p p' =
  List.exists
    (fun (u, w) ->
      List.exists
        (fun (u', w') ->
          (w || w')
          &&
          match (u, u') with
          | `View v, `View v' -> overlap v v'
          | (`View _ | `Words), (`View _ | `Words) -> false
          | `Slot, _ | _, `Slot -> true)
        (uses c p'))
    (uses c p)

(* The layout the order law states: a fill or a launch on queue 0, the first
   that runs it; words on queue 0, the first that runs launches and words; a
   copy on the queue of the last earlier part it follows, else on the copy
   engine, queue 1; and per other queue, the last earlier part it follows. A
   part follows the earlier parts that may share memory with it, and words every
   earlier part. *)
let expected_layout c =
  let parts = Array.of_list c.parts in
  let queues = Array.make (Array.length parts) 0 in
  Array.mapi
    (fun j p ->
      let follows i = p = Words || conflict c parts.(i) p in
      let earlier = List.init j Fun.id in
      let q =
        match p with
        | Fill _ | Launch _ | Words -> 0
        | Copy _ -> (
            match List.filter follows (List.rev earlier) with
            | i :: _ -> queues.(i)
            | [] -> 1)
      in
      queues.(j) <- q;
      let after =
        List.filter_map
          (fun q' ->
            if q' = q then None
            else
              List.fold_left
                (fun last i ->
                  if queues.(i) = q' && follows i then Some i else last)
                None earlier)
          [ 0; 1 ]
      in
      (q, List.sort Int.compare after))
    parts
  |> Array.to_list

let order_law c =
  let d, p = P.open_ ~host_visible:c.host_visible (fresh "order") in
  let host =
    B.view (on d (B.create Rig.host page_bytes)) ~first:0 ~length:alloc_bytes
  in
  let allocs = [| B.create d alloc_bytes; B.create d alloc_bytes; host |] in
  let buffer v = B.view allocs.(v.alloc) ~first:v.first ~length:v.length in
  let slots = slot_order c in
  let index k =
    let rec go i = function
      | [] -> assert false
      | k' :: rest -> if k = k' then i else go (i + 1) rest
    in
    go 0 slots
  in
  let ref_of i k =
    let slot = index k and at = 8 * i in
    match c.kinds.(k) with
    | Read -> Sub.read slot ~at
    | Write -> Sub.write slot ~at
    | Unordered -> Sub.unordered slot ~at
  in
  let side = function
    | View v -> Sub.Buffer (buffer v)
    | Slot k -> Sub.Slot (index k)
  in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")
  in
  let main = require_some (Rig.Image.resolve image "main") in
  let part = function
    | Copy (src, dst) -> Sub.copy ~src:(side src) ~dst:(side dst)
    | Fill ks ->
        Sub.fill
          ~refs:(Array.of_list (List.mapi ref_of ks))
          Support.noop
          ~params:(8 * List.length ks)
    | Launch (ks, entry) ->
        let n = List.length ks in
        let entries = if entry then [ Sub.entry main ~at:(8 * n) ] else [] in
        Sub.launch
          ~refs:(Array.of_list (List.mapi ref_of ks @ entries))
          main
          ~params:(8 * (n + List.length entries))
    | Words -> Sub.words (B.create Rig.host 8)
  in
  let fixed = List.map (fun (v, a) -> (buffer v, a)) c.fixed in
  let s = Sub.make ~fixed d (Array.of_list (List.map part c.parts)) in
  let buffers = Array.of_list (List.map (fun _ -> B.create d 8) slots) in
  (* Each launch runs one group of one thread, which Polled's [main] spends
     doing nothing, and a fill reads no value. *)
  let one_thread =
    { Support.zero with groups = (1, 1, 1); threads = (1, 1, 1) }
  in
  let args =
    List.filter_map
      (function
        | Launch _ -> Some one_thread
        | Fill _ -> Some Support.zero
        | Copy _ | Words -> None)
      c.parts
  in
  Rig.Point.wait (submit s ~buffers ~args:(Array.of_list args));
  let expected = expected_layout c in
  let copy_after_fill =
    List.exists (fun (q, after) -> q = 1 && after <> []) expected
  in
  cover "a copy on the copy engine after a fill" copy_after_fill;
  cover "a copy on a fill's queue"
    (List.exists2
       (fun p (q, _) ->
         match p with Copy _ -> q = 0 | Fill _ | Launch _ | Words -> false)
       c.parts expected);
  let parts = Array.of_list c.parts in
  cover "words after a part of another queue"
    (List.exists2
       (fun p (_, after) -> p = Words && after <> [])
       c.parts expected);
  cover "a copy concurrent with earlier words"
    (List.exists
       (fun (j, (q, after)) ->
         q = 1
         && List.exists
              (fun i -> parts.(i) = Words && not (List.mem i after))
              (List.init j Fun.id))
       (List.mapi (fun j x -> (j, x)) expected));
  cover "an unordered slot"
    (List.exists
       (function
         | Fill ks | Launch (ks, _) ->
             List.exists (fun k -> c.kinds.(k) = Unordered) ks
         | Copy _ | Words -> false)
       c.parts);
  cover "a copy with a slot end"
    (List.exists
       (function
         | Copy (Slot _, _) | Copy (_, Slot _) -> true
         | Copy _ | Fill _ | Launch _ | Words -> false)
       c.parts);
  cover "a launch with an entry ref"
    (List.exists
       (function
         | Launch (_, entry) -> entry | Copy _ | Fill _ | Words -> false)
       c.parts);
  let views =
    let side = function View v -> [ v ] | Slot _ -> [] in
    List.concat_map
      (function
        | Copy (s, d) -> side s @ side d | Fill _ | Launch _ | Words -> [])
      c.parts
    @ List.map fst c.fixed
  in
  cover "two views that share some bytes and differ"
    (List.exists
       (fun v -> List.exists (fun v' -> v <> v' && share_byte v v') views)
       views);
  cover "two views that share a word and no byte"
    (List.exists
       (fun v ->
         List.exists (fun v' -> overlap v v' && not (share_byte v v')) views)
       views);
  (* Unaddressed, each of the device's allocations is a space of its own, cut at
     the bounds of the views of it rounded out to words: [m] ranges between
     them. *)
  let ranges a =
    let w = Rig.word in
    let bounds =
      List.concat_map
        (fun v ->
          if v.alloc = a && v.length > 0 then
            [ v.first / w * w; (v.first + v.length + w - 1) / w * w ]
          else [])
        views
    in
    List.length (List.sort_uniq Int.compare bounds) - 1
  in
  let spaces = if c.host_visible then [] else [ ranges 0; ranges 1 ] in
  let pow2 m = m > 0 && m land (m - 1) = 0 in
  cover "a space of one range" (List.mem 1 spaces);
  cover "a space of a power of 2 ranges, past 1"
    (List.exists (fun m -> m > 1 && pow2 m) spaces);
  cover "a space of one past a power of 2 ranges, past 2"
    (List.exists (fun m -> m > 2 && pow2 (m - 1)) spaces);
  equal
    (list (pair int (list int)))
    ~msg:"each part's queue and after" expected (P.layout p);
  ignore (Sys.opaque_identity allocs)

(* A copy on the copy engine after a fill, which drawn cases give in about one
   in 40: a fill of no slot, then a copy that shares no memory with it, so on
   the copy engine, then a copy after both, which writes the fixed memory the
   fill reads. *)
let order_examples =
  let v alloc : view = { alloc; first = 0; length = 8 } in
  [
    ({
       host_visible = true;
       kinds = [| Unordered; Unordered; Unordered |];
       parts =
         [
           Fill []; Copy (View (v 0), View (v 1)); Copy (View (v 1), View (v 2));
         ];
       fixed = [ (v 2, Rig.Read) ];
     }
      : order_case);
  ]

(* Words, whose effect rig cannot see, follow every earlier part; a later part
   follows them by running on their queue or by sharing memory with them.
   Polled's COMPUTE:0 (queue 0) runs launches, fills and words, COPY:0 (queue 1)
   is its copy engine: - 0, a copy a -> b: on the copy engine; - 1, words
   reading ws: on COMPUTE:0, the queue of launches, after 0, though they share
   no memory; - 2, a fill: on COMPUTE:0, after the words by that queue's order;
   - 3, a copy c -> e: on the copy engine, concurrent with the words; - 4, a
   copy e -> ws: it follows 3 by e, and the words by ws, so it runs on 3's queue
   after the words. *)
let test_words_order () =
  let d, p = P.open_ (fresh "words-order") in
  let ws = B.create Rig.host page_bytes in
  let a = B.create d 8 and b = B.create d 8 in
  let c = B.create d page_bytes and e = B.create d page_bytes in
  let copy src dst = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  let parts =
    [|
      copy a b;
      Sub.words ws;
      Sub.fill Support.noop ~params:0;
      copy c e;
      copy e (on d ws);
    |]
  in
  Rig.Point.wait (submit (Sub.make d parts) ~args:one);
  equal
    (list (pair int (list int)))
    [ (1, []); (0, [ 0 ]); (0, []); (1, []); (1, [ 1 ]) ]
    (P.layout p)

(* Memory *)

let pp_access ppf = function
  | Rig.Read -> Format.pp_print_string ppf "Read"
  | Rig.Read_write -> Format.pp_print_string ppf "Read_write"

let access = Gen.of_list ~pp:pp_access [ Rig.Read; Rig.Read_write ]

(* Two devices and one memory *)

(* Each value of the API is two devices and one memory of the first. Devices go
   back to a pool when their program ends, and the next program counts their
   values from where they stood, once their earlier work ran: rig keeps a record
   of every device for the process's life, so a run opens a few devices rather
   than two per program. *)
type system = { devices : (Rig.t * P.t) array; base : int array; m : B.t }

let pool = Mutex.create ()
let free = ref []
let opened = Atomic.make 0

let take () =
  match
    Mutex.protect pool (fun () ->
        match !free with
        | ds :: rest ->
            free := rest;
            Some ds
        | [] -> None)
  with
  | Some ds -> ds
  | None ->
      let n = Atomic.fetch_and_add opened 1 in
      Array.init 2 (fun i -> P.open_ (Printf.sprintf "stamps:%d-%d" n i))

let give t = Mutex.protect pool (fun () -> free := t.devices :: !free)
let device = Gen.of_list ~pp:Format.pp_print_int [ 0; 1 ]
let other i = 1 - i

(* The model: the values each device assigned, and each device's last write of
   the memory and last use of it, 0 for none. *)
type model = { values : int array; writes : int array; uses : int array }

let pp_model ppf r =
  Format.fprintf ppf "values %d %d, writes %d %d, uses %d %d" r.values.(0)
    r.values.(1) r.writes.(0) r.writes.(1) r.uses.(0) r.uses.(1)

let memory = abstract ~pp:pp_model ~release:give "m"
let signaled t i = Rig.signaled (fst t.devices.(i)) - t.base.(i)

let open_model () =
  { values = [| 0; 0 |]; writes = [| 0; 0 |]; uses = [| 0; 0 |] }

let open_system () =
  let devices = take () in
  let base =
    Array.map
      (fun (d, _) ->
        let p = Rig.submit (Sub.make d [||]) ~buffers:[||] ~args:[||] in
        Rig.Point.wait p;
        Rig.Point.value p)
      devices
  in
  { devices; base; m = B.create (fst devices.(0)) 64 }

(* A submit on device [i] that reads or writes the memory: its value, and the
   other device's word once the submit returned. Writing waits for the other
   device's every use, reading for its last write. *)
let submit_system i access t =
  let d = fst t.devices.(i) in
  let use : Rig.access -> _ = function
    | Read -> Sub.read
    | Read_write -> Sub.write
  in
  let s = Sub.make d [| Support.touching [| use access |] |] in
  let m = require_some (B.borrow d t.m) in
  let v = Rig.Point.value (submit ~args:one s ~buffers:[| m |]) - t.base.(i) in
  (v, signaled t (other i))

let assign i access r v =
  r.values.(i) <- v;
  r.uses.(i) <- v;
  if access = Rig.Read_write then r.writes.(i) <- v

let submit_model i access r = function
  | Error e -> raise e
  | Ok (v, seen) ->
      let j = other i in
      equal ~msg:"value" int (r.values.(i) + 1) v;
      (match access with
      | Rig.Read ->
          at_least ~msg:"the other device's write" int ~than:r.writes.(j) seen
      | Rig.Read_write ->
          at_least ~msg:"the other device's use" int ~than:r.uses.(j) seen);
      assign i access r v

(* A write that another domain's write may race: the model states only its
   value, since two concurrent writes need not follow each other. The calls
   after both follow both. *)
let racing_model i r = function
  | Error e -> raise e
  | Ok (v, _) ->
      equal ~msg:"value" int (r.values.(i) + 1) v;
      assign i Rig.Read_write r v

(* A host access waits for each device's last write, or for every use. *)
let wait_system access t =
  B.wait t.m access;
  (signaled t 0, signaled t 1)

let wait_model access r = function
  | Error e -> raise e
  | Ok (s0, s1) -> (
      match access with
      | Rig.Read ->
          at_least ~msg:"device 0's write" int ~than:r.writes.(0) s0;
          at_least ~msg:"device 1's write" int ~than:r.writes.(1) s1
      | Rig.Read_write ->
          at_least ~msg:"device 0's use" int ~than:r.uses.(0) s0;
          at_least ~msg:"device 1's use" int ~than:r.uses.(1) s1)

(* A copy of the memory into a fresh buffer of device [i] reads it: it waits for
   each device's last write. *)
let copy_system i t =
  let dst = B.create (fst t.devices.(i)) 64 in
  B.copy ~src:t.m ~dst;
  (signaled t 0, signaled t 1)

let open_command =
  command "open" (Gen.unit @-> makes memory) open_model open_system

let wait_command =
  command "wait"
    (access @-> memory ^-> judges (pair int int))
    wait_model wait_system

let copy_command =
  command "copy"
    (device @-> memory ^-> judges (pair int int))
    (fun _ -> wait_model Rig.Read)
    copy_system

let run_command =
  command "run"
    (device @-> memory ^-> returns unit)
    (fun _ _ -> ())
    (fun i t -> ignore (P.run (snd t.devices.(i))))

let commands =
  [
    open_command;
    command "read"
      (device @-> memory ^-> judges (pair int int))
      (fun i -> submit_model i Rig.Read)
      (fun i -> submit_system i Rig.Read);
    (* A write excludes every other submit naming the memory, as its caller's
       exclusive claim does: its [pre] keeps it out of the parallel calls. *)
    command "write"
      ~pre:(fun _ _ -> true)
      (device @-> memory ^-> judges (pair int int))
      (fun i -> submit_model i Rig.Read_write)
      (fun i -> submit_system i Rig.Read_write);
    wait_command;
    copy_command;
    run_command;
  ]

(* Writes in any place of a program, with no submitted read: a read collects its
   stamps and raises them at two instants, which no one-instant model of a
   submit beside a racing write explains. *)
let racing_commands =
  [
    open_command;
    command "racing write"
      (device @-> memory ^-> judges (pair int int))
      racing_model
      (fun i -> submit_system i Rig.Read_write);
    wait_command;
    copy_command;
    run_command;
  ]

(* Two writers *)

(* Two writes of one memory on devices [x] and [y] that do not follow each
   other: [x]'s write collects its stamps, then blocks in its hand-over, its
   queue full, while [y]'s write submits and returns. Each raises the stamps
   unaware of the other, [x]'s last. Device [x]'s queue then holds its write
   unrun, and [y]'s holds its own. *)
type racing = {
  x : Rig.t;
  px : P.t;
  y : Rig.t;
  m : B.t;
  a : int; (* x's write's value *)
  b : int; (* y's write's value *)
}

let racing_writes name =
  let x, px = P.open_ ~capacity:1 ~may_block:true (name ^ "-x") in
  let y, _ = P.open_ (name ^ "-y") in
  let m = B.create x 64 in
  let my = require_some (B.borrow y m) in
  let write d b =
    let s = Sub.make d [| Support.touching [| Sub.write |] |] in
    submit ~args:one s ~buffers:[| b |]
  in
  let first = submit ~args:one (Sub.make x [| Support.touching [||] |]) in
  let a = Domain.spawn (fun () -> write x m) in
  Support.await "x's write blocked" (fun () -> P.blocked px = 1);
  let b = write y my in
  Rig.Point.wait first;
  let a = Domain.join a in
  { x; px; y; m; a = Rig.Point.value a; b = Rig.Point.value b }

let test_racing_host_read () =
  let r = racing_writes "stamps:host-read" in
  B.wait r.m Rig.Read;
  at_least ~msg:"x's word" int ~than:r.a (Rig.signaled r.x);
  at_least ~msg:"y's word" int ~than:r.b (Rig.signaled r.y)

let test_racing_submit_read () =
  let r = racing_writes "stamps:submit-read" in
  ignore (P.run r.px);
  let read = Sub.make r.x [| Support.touching [| Sub.read |] |] in
  ignore (submit ~args:one read ~buffers:[| r.m |]);
  at_least ~msg:"y's word once x's read returned" int ~than:r.b
    (Rig.signaled r.y)

let test_racing_claim () =
  let r = racing_writes "stamps:claim" in
  ignore (P.run r.px);
  let answer = Testable.make ~pp:Support.Reader.pp_answer ~equal:( = ) in
  let got = Support.Reader.claim r.m Rig.Read in
  Support.Reader.release r.m;
  less ~msg:"y's word" int ~than:r.b (Rig.signaled r.y);
  equal ~msg:"a read claim with only x's write run" answer Support.Reader.Wait
    got

(* A submit whose hand-over fails has assigned its value: the stamps of the
   memory it writes name it, so a read of that memory raises the loss, and one
   of memory it only read does not. *)
let test_failed_names () =
  let d, p = P.open_ (fresh "failed-names") in
  let w = B.create Rig.host page_bytes and r = B.create Rig.host page_bytes in
  let s = Sub.make d [| Support.touching [| Sub.read; Sub.write |] |] in
  let opened = Rig.submitted d in
  P.fail p;
  let lost = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false in
  raises_match ~msg:"the submit" lost (fun () ->
      submit ~args:one s ~buffers:[| on d r; on d w |]);
  equal int ~msg:"its value" (opened + 1) (Rig.submitted d);
  raises_match ~msg:"a read of its write" lost (fun () -> B.wait w Rig.Read);
  B.wait r Rig.Read;
  raises_match ~msg:"a write of its read" lost (fun () ->
      B.wait r Rig.Read_write)

(* A read of memory another device wrote waits for that write: on the host,
   since a software device waits on no other device in its queue. *)
let test_read_waits () =
  let producer, pp = P.open_ (fresh "producer") in
  let consumer = software "consumer" in
  let b = B.create producer page_bytes in
  ignore (touch [| Sub.write |] producer [| b |]);
  equal int 1 (P.queued pp);
  ignore (touch [| Sub.read |] consumer [| on consumer b |]);
  equal int 0 (P.queued pp)

(* A part's buffers are ordered as a submit's are: a copy waits for another
   device's write of its source and for another device's read of its
   destination. *)
let test_part_points () =
  let d, _ = P.open_ (fresh "parts") in
  let e, pe = P.open_ (fresh "parts-other") in
  let src = B.create d 64 and dst = B.create d 64 in
  let copy = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  let s = Sub.make d [| copy |] in
  ignore (touch [| Sub.write |] e [| on e src |]);
  ignore (submit s);
  equal ~msg:"after a write of the source" int 0 (P.queued pe);
  ignore (touch [| Sub.read |] e [| on e dst |]);
  ignore (submit s);
  equal ~msg:"after a read of the destination" int 0 (P.queued pe)

(* A launch of Polled's "fill" over 2 x 2 groups, whose group [i] stores [100 +
   i] as the 64-bit word [i] of the buffer it writes: once a host wait for a
   read returned, the host reads every group's word with no call of rig's. *)
let test_write_seen () =
  let d, _ = P.open_ (fresh "write-seen") in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")
  in
  let fill = require_some (Rig.Image.resolve image "fill") in
  let out = B.create d 32 in
  let s =
    Sub.make d [| Sub.launch ~refs:[| Sub.write 0 ~at:0 |] fill ~params:16 |]
  in
  let params = Bytes.make 16 '\000' in
  Bytes.set_int64_le params 8 100L;
  let args =
    [| { Sub.groups = (2, 2, 1); threads = (1, 1, 1); shared = 0; params } |]
  in
  ignore (submit ~buffers:[| out |] ~args s);
  B.wait out Read;
  let at = Support.Reader.host out in
  equal ~msg:"each group's word" (list int) [ 100; 101; 102; 103 ]
    (List.init 4 (fun i -> Support.load (at + (8 * i))))

(* Submission.unordered *)

(* An [Unordered] buffer orders nothing: a submit that uses memory another
   device's queued work writes waits for none of it, that device's later write
   does not wait for the use, and neither does a wait of the host. *)
let test_unordered () =
  let a, pa = P.open_ (fresh "unordered-writer") in
  let b, pb = P.open_ (fresh "unordered-user") in
  let x = B.create a page_bytes in
  let write () = ignore (touch [| Sub.write |] a [| x |]) in
  write ();
  ignore (touch [| Sub.unordered |] b [| on b x |]);
  equal ~msg:"the writer's work, after the use" int 1 (P.queued pa);
  write ();
  equal ~msg:"the use, after the next write" int 1 (P.queued pb);
  B.wait x Rig.Read_write;
  equal ~msg:"the use, after the host's wait" int 1 (P.queued pb);
  ignore (P.run pb)

(* An [Unordered] buffer stays reachable until the work is done: memory of one
   device that only such a submit on another named returns once that device ran
   the work, not before. *)
let test_unordered_keeps () =
  let a, pa = P.open_ (fresh "unordered-keeps-owner") in
  let b, pb = P.open_ (fresh "unordered-keeps-user") in
  (fun () ->
    let x = B.create a page_bytes in
    ignore (touch [| Sub.unordered |] b [| on b x |]))
    ();
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create a 0);
    Rig.free_cache a
  in
  let frees () = List.length (List.filter (( = ) "free") (P.log pa)) in
  drain ();
  equal ~msg:"while the work is queued" int 0 (frees ());
  ignore (P.run pb);
  drain ();
  equal ~msg:"once it ran" int 1 (frees ())

(* A collected buffer's queued write, through [use], never reaches a buffer made
   later on its memory, whichever buffer's access is unordered. *)
let[@inline never] [@local never] dropped_bump d use =
  let b = B.create d page_bytes in
  Support.store (Support.Reader.host b) 0;
  let bump = Sub.fill ~refs:[| use 0 ~at:0 |] Support.bump ~params:8 in
  submit (Sub.make d [| bump |]) ~buffers:[| b |] ~args:one

let test_cached_unordered_host () =
  let d, p = P.open_ (fresh "cached-unordered-host") in
  let old = dropped_bump d Sub.unordered in
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"the earlier work is pending" int 1 (P.queued p);
  let b = B.create d page_bytes in
  B.wait b Rig.Read_write;
  Support.store (Support.Reader.host b) 41;
  Rig.Point.wait old;
  equal ~msg:"the new buffer after the old work finishes" int 41 (word b)

let test_cached_unordered_foreign (name, access) =
  let a, pa = P.open_ (fresh ("cached-unordered-owner:" ^ name)) in
  let d, _ = P.open_ (fresh ("cached-unordered-user:" ^ name)) in
  let old = dropped_bump a access in
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"the earlier work is pending" int 1 (P.queued pa);
  let b = B.create a page_bytes in
  let poke =
    Sub.fill ~refs:[| Sub.unordered 0 ~at:0 |] Support.poke ~params:16
  in
  let s = Sub.make d [| poke |] in
  let params = Bytes.make 16 '\000' in
  Bytes.set_int64_le params 8 41L;
  let args = [| { Support.zero with params } |] in
  Rig.Point.wait (submit ~buffers:[| on d b |] ~args s);
  Rig.Point.wait old;
  equal ~msg:"the new buffer after the old work finishes" int 41 (word b)

(* Words *)

(* rig.mli: "[word] is [4], the bytes of a word." *)
let test_word () = equal ~msg:"Rig.word" int 4 Rig.word

let tests =
  [
    group "device order"
      [
        test "a value's work follows the previous value's, on every queue"
          test_device_order;
      ];
    group "parts"
      [
        prop ~count:300 ~examples:order_examples
          "a part follows the earlier parts that may share memory with it, on \
           its queue or by its after"
          (Gen.with_pp pp_order_case gen_order_case)
          order_law;
        test "words follow every earlier part, and later parts of their queue"
          test_words_order;
      ];
    group "memory"
      [
        stateful "submits and waits follow the memory's stamps" commands;
        stateful ~count:30 ~domains:2
          "stamps raised from two domains are their maxima" commands;
        stateful ~count:30 ~domains:2
          "accesses after two racing writes follow both" racing_commands;
        test "a host read after two racing writes follows both"
          test_racing_host_read;
        test "a submitted read after two racing writes follows both"
          test_racing_submit_read;
        test "a read claim after two racing writes waits for both"
          test_racing_claim;
        test "a submit its driver fails to take names its memory"
          test_failed_names;
        test "a read waits for another device's write" test_read_waits;
        test "a part's buffers wait as a submit's do" test_part_points;
        test "a read of a buffer waits for the launch that writes it"
          test_write_seen;
      ];
    group "Submission.unordered"
      [
        test "an Unordered buffer orders no work" test_unordered;
        test "an Unordered buffer stays reachable until its work is done"
          test_unordered_keeps;
        test
          "an unordered write of a collected buffer never reaches a new \
           buffer's host write"
          test_cached_unordered_host;
        cases
          "a write of a collected buffer never reaches a new buffer's \
           unordered write"
          ~id:fst
          [
            ("after ordered work", Sub.write);
            ("after Unordered work", Sub.unordered);
          ]
          test_cached_unordered_foreign;
      ];
    group "word" [ test "a word is 4 bytes" test_word ];
  ]

let () = exit (run "rig.stamps" tests)
