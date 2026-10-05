(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Environment *)

(* Numbers are read with Python's number grammar, restricted to ASCII. *)

let is_space c = (c >= '\t' && c <= '\r') || c = ' '
let is_digit c = c >= '0' && c <= '9'

let strip s =
  let i = ref 0 and j = ref (String.length s) in
  while !i < !j && is_space s.[!i] do
    incr i
  done;
  while !j > !i && is_space s.[!j - 1] do
    decr j
  done;
  String.sub s !i (!j - !i)

let sign s i =
  if i < String.length s && (s.[i] = '+' || s.[i] = '-') then i + 1 else i

(* The end of the digits starting at [i], an underscore being allowed between
   two digits; [i] if no digit starts there. OCaml's parsers accept an
   underscore anywhere after the first digit, Python only between two. *)
let digits s i =
  let n = String.length s in
  let rec after_digit j =
    if j < n && is_digit s.[j] then after_digit (j + 1)
    else if j + 1 < n && s.[j] = '_' && is_digit s.[j + 1] then
      after_digit (j + 2)
    else j
  in
  if i < n && is_digit s.[i] then after_digit (i + 1) else i

(* [is_int] and [is_float] refuse what OCaml's parsers accept and Python does
   not: misplaced underscores, base prefixes, hexadecimal floats and [nan(...)].
   A missing digit run is left to the parsers, which refuse it. *)

let is_int t = digits t (sign t 0) = String.length t

let is_float t =
  let n = String.length t in
  let i = sign t 0 in
  let j = digits t i in
  let k = if j < n && t.[j] = '.' then digits t (j + 1) else j in
  let exponent_end =
    if k < n && (t.[k] = 'e' || t.[k] = 'E') then digits t (sign t (k + 1))
    else k
  in
  List.mem
    (String.lowercase_ascii (String.sub t i (n - i)))
    [ "inf"; "infinity"; "nan" ]
  || exponent_end = n

let parse_int s =
  let t = strip s in
  match if is_int t then int_of_string_opt t else None with
  | Some n -> Ok n
  | None -> Error (Printf.sprintf "%S is not an integer" s)

let parse_float s =
  let t = strip s in
  match if is_float t then float_of_string_opt t else None with
  | Some x -> Ok x
  | None -> Error (Printf.sprintf "%S is not a number" s)

let read key parse default =
  match Sys.getenv_opt key with
  | None -> default
  | Some s -> (
      match parse s with
      | Ok x -> x
      | Error e -> invalid_arg (Printf.sprintf "%s: %s" key e))

(* Declarations

   Each setting has a name no other one has. Those whose reach is [Output] are
   recorded, sorted by name, with the text of their current value, on which the
   caches of programs, schedules and searches are keyed. *)

type reach = Output | Process

module Names = Set.Make (String)

let declared = Atomic.make Names.empty
let outputs = Atomic.make []

let rec declare_name key =
  let names = Atomic.get declared in
  if Names.mem key names then
    invalid_arg (Printf.sprintf "%s is already declared" key);
  if not (Atomic.compare_and_set declared names (Names.add key names)) then
    declare_name key

let by_key (k, _) (k', _) = compare k k'

(* Each domain holds, in an atomic, the entries of [shaping] with the
   declarations they show, so that reading them allocates nothing while nothing
   changes. Each change is a compare-and-set of the whole, so the changes of
   systhreads sharing a domain compose: a setting replaces its own entry with
   the text of its value as it is once it holds it. Entries that show other
   declarations than there are, on a domain that a later declaration did not
   reach, are rendered anew from every value. A spawned domain starts with a
   copy of its parent's entries, as with its values. *)
let entries =
  Domain.DLS.new_key
    ~split_from_parent:(fun e -> Atomic.make (Atomic.get e))
    (fun () -> Atomic.make ([], []))

let render shown = List.map (fun (key, show) -> (key, show ())) shown

(* [update f] makes the calling domain's entries [f e], or renders them anew if
   they show other declarations than there are. *)
let rec update f =
  let a = Domain.DLS.get entries in
  let ((shown, e) as current) = Atomic.get a in
  let declared = Atomic.get outputs in
  let next =
    if shown == declared then (shown, f e) else (declared, render declared)
  in
  if not (Atomic.compare_and_set a current next) then update f

(* A declaration merges its entry into the declaring domain's. *)
let rec record key show =
  let l = Atomic.get outputs in
  let l' = List.merge by_key [ (key, show) ] l in
  if not (Atomic.compare_and_set outputs l l') then record key show
  else
    let a = Domain.DLS.get entries in
    let ((shown, e) as current) = Atomic.get a in
    if shown == l then
      let merged = List.merge by_key [ (key, show ()) ] e in
      ignore (Atomic.compare_and_set a current (l', merged))

let rec shaping () =
  let shown, e = Atomic.get (Domain.DLS.get entries) in
  if shown == Atomic.get outputs then e
  else (
    update Fun.id;
    shaping ())

(* Settings *)

(* A domain starts with the values of the domain that spawns it. *)
type 'a t = {
  key : string;
  reach : reach;
  show : 'a -> string;
  value : 'a Domain.DLS.key;
}

let v ~reach ~show key x =
  declare_name key;
  let value = Domain.DLS.new_key ~split_from_parent:Fun.id (fun () -> x) in
  (match reach with
  | Output -> record key (fun () -> show (Domain.DLS.get value))
  | Process -> ());
  { key; reach; show; value }

let int ~reach key default =
  v ~reach ~show:string_of_int key (read key parse_int default)

let bool ~reach key default =
  let n = read key parse_int (Bool.to_int default) in
  v ~reach ~show:string_of_bool key (n <> 0)

(* Numbers are shown in hexadecimal, each with a text of its own. *)
let float ~reach key default =
  v ~reach ~show:(Printf.sprintf "%h") key (read key parse_float default)

let string ~reach key default =
  v ~reach ~show:Fun.id key (read key Result.ok default)

let int_option ~reach key =
  let parse s = Result.map Option.some (parse_int s) in
  v ~reach
    ~show:(Option.fold ~none:"" ~some:string_of_int)
    key (read key parse None)

let key v = v.key
let value v = Domain.DLS.get v.value

let set v x =
  Domain.DLS.set v.value x;
  match v.reach with
  | Process -> ()
  | Output ->
      let entry ((key, _) as e) =
        if String.equal key v.key then (key, v.show (value v)) else e
      in
      update (List.map entry)

type binding = B : 'a t * 'a -> binding

let context bindings f =
  let swap saved (B (v, x)) =
    let previous = value v in
    set v x;
    B (v, previous) :: saved
  in
  let saved = List.fold_left swap [] bindings in
  let restore () = List.iter (fun (B (v, x)) -> set v x) saved in
  Fun.protect ~finally:restore f

(* Tolk's settings *)

let debug = int ~reach:Process "DEBUG" 0
let beam = int ~reach:Output "BEAM" 0
let jitbeam = int_option ~reach:Output "JITBEAM"
let noopt = bool ~reach:Output "NOOPT" false
let no_color = bool ~reach:Process "NO_COLOR" false
let use_tc = int ~reach:Output "TC" 1
let tc_select = int ~reach:Output "TC_SELECT" (-1)
let tc_opt = int ~reach:Output "TC_OPT" 0
let beam_tc_opt = int ~reach:Output "BEAM_TC_OPT" 2
let tc_min_globals = int ~reach:Output "TC_MIN_GLOBALS" 0
let transcendental = int ~reach:Output "TRANSCENDENTAL" 1
let split_reduceop = bool ~reach:Output "SPLIT_REDUCEOP" true
let no_memory_planner = bool ~reach:Output "NO_MEMORY_PLANNER" false
let ring = int ~reach:Output "RING" 1
let all2all = int ~reach:Output "ALL2ALL" 0
let allreduce_cast = bool ~reach:Output "ALLREDUCE_CAST" true
let allreduce_node_ndevs = int ~reach:Output "ALLREDUCE_NODE_NDEVS" 0
let cachelevel = int ~reach:Process "CACHELEVEL" 2
let ignore_beam_cache = bool ~reach:Process "IGNORE_BEAM_CACHE" false
let disable_fast_idiv = bool ~reach:Output "DISABLE_FAST_IDIV" true
let max_kernel_buffers = int ~reach:Output "MAX_KERNEL_BUFFERS" 0

let emulated_dtypes =
  let names = String.split_on_char ',' (read "EMULATED_DTYPES" Result.ok "") in
  v ~reach:Output ~show:(String.concat ",") "EMULATED_DTYPES"
    (List.filter (fun x -> x <> "") names)

let default_float = string ~reach:Output "DEFAULT_FLOAT" "float32"
let default_int = string ~reach:Output "DEFAULT_INT" "int32"

(* A container's CPU quota lives in the cgroup v2 file as "QUOTA PERIOD", or
   "max PERIOD" when unlimited. *)
let cpu_count =
  let count = Domain.recommended_domain_count () in
  match
    In_channel.with_open_text "/sys/fs/cgroup/cpu.max" In_channel.input_all
  with
  | exception Sys_error _ -> count
  | contents -> (
      match String.split_on_char ' ' (String.trim contents) with
      | [ quota; period ] when quota <> "max" -> (
          match (parse_int quota, parse_int period) with
          | Ok quota, Ok period when period <> 0 ->
              min count (max 1 (quota / period))
          | _ -> count)
      | _ -> count)

let parallel = int ~reach:Process "PARALLEL" cpu_count
let spec = int ~reach:Process "SPEC" 1
let check_oob = bool ~reach:Process "CHECK_OOB" false
let debug_rangeify = bool ~reach:Process "DEBUG_RANGEIFY" false
let tuple_order = bool ~reach:Output "TUPLE_ORDER" true
let ccache = bool ~reach:Process "CCACHE" true
let allow_tf32 = bool ~reach:Output "ALLOW_TF32" false
let scache = int ~reach:Process "SCACHE" 2
let disallow_broadcast = bool ~reach:Process "DISALLOW_BROADCAST" false
