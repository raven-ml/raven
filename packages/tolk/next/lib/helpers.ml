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
   two digits; [i] if no digit starts there. *)
let digits s i =
  let n = String.length s in
  let rec after_digit j =
    if j < n && is_digit s.[j] then after_digit (j + 1)
    else if j + 1 < n && s.[j] = '_' && is_digit s.[j + 1] then
      after_digit (j + 2)
    else j
  in
  if i < n && is_digit s.[i] then after_digit (i + 1) else i

let is_int t =
  let i = sign t 0 in
  let j = digits t i in
  j > i && j = String.length t

let is_float t =
  let n = String.length t in
  let i = sign t 0 in
  let j = digits t i in
  let k = if j < n && t.[j] = '.' then digits t (j + 1) else j in
  let mantissa = j > i || k > j + 1 in
  let exponent_end =
    if k < n && (t.[k] = 'e' || t.[k] = 'E') then
      let m = sign t (k + 1) in
      let e = digits t m in
      if e > m then e else k
    else k
  in
  List.mem
    (String.lowercase_ascii (String.sub t i (n - i)))
    [ "inf"; "infinity"; "nan" ]
  || (mantissa && exponent_end = n)

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

(* [read] remembered per key and default, so that a program sees one value of
   each variable however its environment changes. Entries are published
   atomically; racing first reads publish once, and all get that value. *)
let memoize parse =
  let memo = Atomic.make [] in
  let rec publish k v =
    let m = Atomic.get memo in
    match List.assoc_opt k m with
    | Some v -> v
    | None ->
        if Atomic.compare_and_set memo m ((k, v) :: m) then v else publish k v
  in
  fun key default ->
    match List.assoc_opt (key, default) (Atomic.get memo) with
    | Some v -> v
    | None -> publish (key, default) (read key parse default)

let getenv = memoize parse_int
let getenv_float = memoize parse_float
let getenv_string = memoize Result.ok

(* Settings *)

module Context_var = struct
  type 'a t = { key : string; mutable value : 'a }

  module Keys = Set.Make (String)

  let declared = Atomic.make Keys.empty

  let rec declare key =
    let keys = Atomic.get declared in
    if Keys.mem key keys then
      invalid_arg (Printf.sprintf "setting %s is already declared" key);
    if not (Atomic.compare_and_set declared keys (Keys.add key keys)) then
      declare key

  let v key value =
    declare key;
    { key; value }

  let int key default = v key (getenv key default)
  let bool key default = v key (getenv key (Bool.to_int default) <> 0)
  let string key default = v key (getenv_string key default)
  let key v = v.key
  let value v = v.value
end

type binding = B : 'a Context_var.t * 'a -> binding

let context bindings f =
  let swap saved (B (v, x)) =
    let previous = Context_var.value v in
    v.Context_var.value <- x;
    B (v, previous) :: saved
  in
  let saved = List.fold_left swap [] bindings in
  let restore () =
    List.iter (fun (B (v, x)) -> v.Context_var.value <- x) saved
  in
  Fun.protect ~finally:restore f

module Target = struct
  type t = {
    device : string;
    renderer : string;
    arch : string;
    interface : string;
    indices : string;
  }

  let empty =
    { device = ""; renderer = ""; arch = ""; interface = ""; indices = "" }

  let parse s =
    let too_many sep s =
      Error (Printf.sprintf "too many '%c' in target string: '%s'" sep s)
    in
    let split =
      match String.split_on_char '+' s with
      | [ prefix; rest ] -> (
          match String.rindex_opt prefix ':' with
          | Some i ->
              Ok
                ( String.sub prefix 0 i,
                  String.sub prefix (i + 1) (String.length prefix - i - 1),
                  rest )
          | None -> Ok (prefix, "", rest))
      | [ _ ] -> Ok ("", "", s)
      | _ -> too_many '+' s
    in
    Result.bind split (fun (interface, indices, s) ->
        let t = { empty with interface; indices }
        and up = String.uppercase_ascii in
        match String.split_on_char ':' s with
        | [ device ] -> Ok { t with device = up device }
        | [ device; renderer ] ->
            Ok { t with device = up device; renderer = up renderer }
        | [ device; renderer; arch ] ->
            Ok { t with device = up device; renderer = up renderer; arch }
        | _ -> too_many ':' s)

  let join fields =
    let s = String.concat ":" fields in
    let n = ref (String.length s) in
    while !n > 0 && s.[!n - 1] = ':' do
      decr n
    done;
    String.sub s 0 !n

  let to_string t =
    let fst = join [ t.interface; t.indices ] in
    (if fst = "" then "" else fst ^ "+") ^ join [ t.device; t.renderer; t.arch ]
end

let parse_targets s =
  let add t acc =
    Result.bind acc (fun ts -> Result.map (fun t -> t :: ts) (Target.parse t))
  in
  List.fold_right add (String.split_on_char ';' s) (Ok [])

let dev =
  match getenv_string "DEV" "" |> parse_targets with
  | Ok targets -> Context_var.v "DEV" targets
  | Error e -> invalid_arg ("DEV: " ^ e)

let target ?(arch = "") device =
  let matches (t : Target.t) = t.device = "" || t.device = device in
  let t =
    Option.value
      (List.find_opt matches (Context_var.value dev))
      ~default:Target.empty
  in
  { t with device; arch = (if t.arch = "" then arch else t.arch) }

let debug = Context_var.int "DEBUG" 0
let beam = Context_var.int "BEAM" 0
let noopt = Context_var.bool "NOOPT" false
let no_color = Context_var.bool "NO_COLOR" false
let use_tc = Context_var.int "TC" 1
let tc_select = Context_var.int "TC_SELECT" (-1)
let tc_opt = Context_var.int "TC_OPT" 0
let tc_min_globals = Context_var.int "TC_MIN_GLOBALS" 0
let transcendental = Context_var.int "TRANSCENDENTAL" 1
let split_reduceop = Context_var.bool "SPLIT_REDUCEOP" true
let no_memory_planner = Context_var.bool "NO_MEMORY_PLANNER" false
let ring = Context_var.int "RING" 1
let all2all = Context_var.int "ALL2ALL" 0
let allreduce_cast = Context_var.bool "ALLREDUCE_CAST" true
let allreduce_node_ndevs = Context_var.int "ALLREDUCE_NODE_NDEVS" 0
let cachelevel = Context_var.int "CACHELEVEL" 2
let ignore_beam_cache = Context_var.bool "IGNORE_BEAM_CACHE" false
let disable_fast_idiv = Context_var.bool "DISABLE_FAST_IDIV" true
let max_kernel_buffers = Context_var.int "MAX_KERNEL_BUFFERS" 0

let emulated_dtypes =
  let names = String.split_on_char ',' (getenv_string "EMULATED_DTYPES" "") in
  Context_var.v "EMULATED_DTYPES" (List.filter (fun x -> x <> "") names)

let default_float = Context_var.string "DEFAULT_FLOAT" "float32"
let default_int = Context_var.string "DEFAULT_INT" "int32"

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

let parallel = Context_var.int "PARALLEL" cpu_count
let spec = Context_var.int "SPEC" 1
let check_oob = Context_var.bool "CHECK_OOB" false
let debug_rangeify = Context_var.bool "DEBUG_RANGEIFY" false
let tuple_order = Context_var.bool "TUPLE_ORDER" true
let ccache = Context_var.bool "CCACHE" true
let allow_tf32 = Context_var.bool "ALLOW_TF32" false
let scache = Context_var.int "SCACHE" 1
let disallow_broadcast = Context_var.bool "DISALLOW_BROADCAST" false

(* Integers and lists *)

let prod l = List.fold_left ( * ) 1 l

let dedup (type a) (module H : Hashtbl.HashedType with type t = a) l =
  let module Seen = Hashtbl.Make (H) in
  let seen = Seen.create 16 in
  let first x =
    let before = Seen.mem seen x in
    Seen.replace seen x ();
    not before
  in
  List.filter first l

let argsort l =
  List.mapi (fun i x -> (x, i)) l
  |> List.stable_sort (fun (x, _) (y, _) -> Int.compare x y)
  |> List.map snd

let all_same equal = function
  | [] -> true
  | first :: _ as l -> List.for_all (fun x -> equal x first) l

let get_single_element = function
  | [ x ] -> x
  | l ->
      invalid_arg
        (Printf.sprintf "get_single_element: %d elements, expected 1"
           (List.length l))

(* Division rounded down; OCaml's [/] rounds towards zero. *)
let fdiv x y =
  let q = x / y in
  if x mod y <> 0 && x < 0 <> (y < 0) then q - 1 else q

let ceildiv num amt = -fdiv num (-amt)
let round_up num amt = fdiv (num + amt - 1) amt * amt
let cdiv x y = if y = 0 then 0 else x / y
let cmod x y = x - (cdiv x y * y)
let floordiv x y = if y = 0 then 0 else fdiv x y
let floormod x y = x - (floordiv x y * y)
let lo32 x = x land 0xFFFF_FFFF
let hi32 x = x asr 32
let data64 x = (hi32 x, lo32 x)
let data64_le x = (lo32 x, hi32 x)

(* Selection *)

(* The similarity of [a] and [b], in [0;1]: twice the length of their matching
   blocks over their total length. A block is the longest common substring, then
   recursively the blocks left and right of it. When [b] has 200 characters or
   more, its characters occurring in more than 1% of it cannot start a block. *)
let similarity a b =
  let la = String.length a and lb = String.length b in
  let positions = Array.make 256 [] in
  for j = lb - 1 downto 0 do
    let c = Char.code b.[j] in
    positions.(c) <- j :: positions.(c)
  done;
  if lb >= 200 then
    Array.iteri
      (fun c js -> if List.length js > (lb / 100) + 1 then positions.(c) <- [])
      positions;
  let longest alo ahi blo bhi =
    let besti = ref alo and bestj = ref blo and size = ref 0 in
    let lengths = ref (Hashtbl.create 8) in
    for i = alo to ahi - 1 do
      let next = Hashtbl.create 8 in
      List.iter
        (fun j ->
          if j >= blo && j < bhi then begin
            let k =
              1 + Option.value (Hashtbl.find_opt !lengths (j - 1)) ~default:0
            in
            Hashtbl.replace next j k;
            if k > !size then (
              besti := i - k + 1;
              bestj := j - k + 1;
              size := k)
          end)
        positions.(Char.code a.[i]);
      lengths := next
    done;
    while !besti > alo && !bestj > blo && a.[!besti - 1] = b.[!bestj - 1] do
      decr besti;
      decr bestj;
      incr size
    done;
    while
      !besti + !size < ahi
      && !bestj + !size < bhi
      && a.[!besti + !size] = b.[!bestj + !size]
    do
      incr size
    done;
    (!besti, !bestj, !size)
  in
  let rec matched alo ahi blo bhi =
    let i, j, k = longest alo ahi blo bhi in
    if k = 0 then 0
    else
      k
      + (if alo < i && blo < j then matched alo i blo j else 0)
      +
      if i + k < ahi && j + k < bhi then matched (i + k) ahi (j + k) bhi else 0
  in
  if la + lb = 0 then 1. else 2. *. float (matched 0 la 0 lb) /. float (la + lb)

(* The most similar of [names] to [query], at least 0.6 similar; the greater
   name wins a tie. *)
let close_match query names =
  let best acc name =
    let score = similarity name query in
    match acc with
    | Some (s, n) when compare (s, n) (score, name) >= 0 -> acc
    | _ when score >= 0.6 -> Some (score, name)
    | _ -> acc
  in
  Option.map snd (List.fold_left best None names)

let select_by_name ~error name query candidates =
  match
    List.filter (fun c -> query = "" || String.equal (name c) query) candidates
  with
  | [] ->
      let hint =
        match close_match query (List.map name candidates) with
        | Some m -> Printf.sprintf ", did you mean: '%s'?" m
        | None -> ""
      in
      Error (error ^ hint)
  | selected -> Ok selected

let select_first_inited ~error candidates =
  let rec select errors = function
    | [] ->
        Error
          (match List.rev errors with
          | [ e ] -> e
          | es -> String.concat "\n" (error :: es))
    | init :: rest -> (
        match init () with
        | Ok _ as ok -> ok
        | Error e -> select (e :: errors) rest)
  in
  select [] candidates

(* Terminal text *)

type color =
  | Black
  | Red
  | Green
  | Yellow
  | Blue
  | Magenta
  | Cyan
  | White
  | Bright_black
  | Bright_red
  | Bright_green
  | Bright_yellow
  | Bright_blue
  | Bright_magenta
  | Bright_cyan
  | Bright_white

let color_code = function
  | Black -> 30
  | Red -> 31
  | Green -> 32
  | Yellow -> 33
  | Blue -> 34
  | Magenta -> 35
  | Cyan -> 36
  | White -> 37
  | Bright_black -> 90
  | Bright_red -> 91
  | Bright_green -> 92
  | Bright_yellow -> 93
  | Bright_blue -> 94
  | Bright_magenta -> 95
  | Bright_cyan -> 96
  | Bright_white -> 97

let colored ?(background = false) color st =
  if Context_var.value no_color then st
  else
    Printf.sprintf "\027[%dm%s\027[0m"
      (color_code color + if background then 10 else 0)
      st

let time_to_str ?(w = 8) t =
  if t > 10. then Printf.sprintf "%*.2fs " w t
  else if t > 10. /. 1e3 then Printf.sprintf "%*.2fms" w (t *. 1e3)
  else Printf.sprintf "%*.2fus" w (t *. 1e6)

let size_to_str s =
  let in_unit d unit = Printf.sprintf "%.2f %s" (float s /. float d) unit in
  if s >= 1 lsl 30 then in_unit (1 lsl 30) "GB"
  else if s >= 1 lsl 20 then in_unit (1 lsl 20) "MB"
  else if s >= 1 lsl 10 then in_unit (1 lsl 10) "KB"
  else Printf.sprintf "%d B" s

let ansistrip s =
  let n = String.length s in
  let b = Buffer.create n in
  (* The end of the escape sequence starting at [i], if one does. *)
  let escape_end i =
    if i + 2 >= n || s.[i] <> '\027' || s.[i + 1] <> '[' then None
    else if s.[i + 2] = 'K' then Some (i + 3)
    else
      let rec find j =
        if j >= n || s.[j] = '\n' then None
        else if s.[j] = 'm' then Some (j + 1)
        else find (j + 1)
      in
      find (i + 2)
  in
  let rec go i =
    if i < n then
      match escape_end i with
      | Some j -> go j
      | None ->
          Buffer.add_char b s.[i];
          go (i + 1)
  in
  go 0;
  Buffer.contents b

let fold_uchars f acc s =
  let rec go i acc =
    if i >= String.length s then acc
    else
      let d = String.get_utf_8_uchar s i in
      go (i + Uchar.utf_decode_length d) (f acc (Uchar.utf_decode_uchar d))
  in
  go 0 acc

let ansilen s = fold_uchars (fun n _ -> n + 1) 0 (ansistrip s)
let ansipad s w = s ^ String.make (max (w - ansilen s) 0) ' '

let is_balanced s =
  let depth = ref 0 in
  String.for_all
    (fun c ->
      if c = '(' then incr depth else if c = ')' then decr depth;
      !depth >= 0)
    s
  && !depth = 0

let strip_parens s =
  let n = String.length s in
  if n >= 2 && s.[0] = '(' && s.[n - 1] = ')' then
    let inner = String.sub s 1 (n - 2) in
    if is_balanced inner then inner else s
  else s

let pluralize st cnt =
  Printf.sprintf "%d %s%s" cnt st (if cnt = 1 then "" else "s")

let is_identifier_char = function
  | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
  | _ -> false

let to_function_name s =
  let b = Buffer.create (String.length s) in
  let add () u =
    match Uchar.to_int u with
    | c when c < 128 && is_identifier_char (Char.chr c) ->
        Buffer.add_char b (Char.chr c)
    | c -> Buffer.add_string b (Printf.sprintf "%02X" c)
  in
  fold_uchars add () (ansistrip s);
  Buffer.contents b

(* Disk cache *)

(* [~/path] expanded as a shell does, and left as is without a home. *)
let expanduser path =
  let home =
    if Sys.win32 then
      match (Sys.getenv_opt "USERPROFILE", Sys.getenv_opt "HOMEPATH") with
      | Some home, _ -> Some home
      | None, Some p ->
          Some (Option.value (Sys.getenv_opt "HOMEDRIVE") ~default:"" ^ p)
      | None, None -> None
    else
      match Sys.getenv_opt "HOME" with
      | Some home -> Some home
      | None -> (
          try Some (Unix.getpwuid (Unix.getuid ())).pw_dir
          with Not_found -> None)
  in
  match home with
  | Some home -> Filename.concat home path
  | None -> Filename.concat "~" path

let cache_dir =
  let base =
    match Sys.getenv_opt "XDG_CACHE_HOME" with
    | Some dir -> dir
    | None ->
        expanduser
          (if Host_config.system = "macosx" then "Library/Caches" else ".cache")
  in
  Filename.concat base "tolk"

let cachedb =
  match Sys.getenv_opt "CACHEDB" with
  | Some dir -> dir
  | None ->
      let dir = Filename.concat cache_dir "cache" in
      if Filename.is_relative dir then Filename.concat (Sys.getcwd ()) dir
      else dir

module Diskcache = struct
  (* Bump whenever what an entry means changes without its key changing: the old
     entries would otherwise answer a different question. *)
  let version = 1
  let digest s = Digest.to_hex (Digest.string s)
  let is_hex c = is_digit c || (c >= 'a' && c <= 'f')

  let starts_with_digest name =
    String.length name >= 32 && String.for_all is_hex (String.sub name 0 32)

  (* A table's directory is named by the digest of its name, so that any name
     makes a valid, bounded file name on every file system, followed by the
     version. *)
  let table_dir table =
    Filename.concat cachedb (Printf.sprintf "%s_%d" (digest table) version)

  let is_table_dir name =
    let n = String.length name in
    starts_with_digest name && n > 33
    && name.[32] = '_'
    && String.for_all is_digit (String.sub name 33 (n - 33))

  (* An entry's file is named by the digest of its key; a write in progress adds
     a suffix to that name. *)
  let is_entry name =
    starts_with_digest name && (String.length name = 32 || name.[32] = '.')

  let entry_path table key = Filename.concat (table_dir table) (digest key)
  let enabled () = Context_var.value cachelevel >= 1

  (* An entry is the lengths of its key and value, a newline, the key and the
     value. Keeping the key tells a digest collision from a hit, and the lengths
     tell a damaged entry from a whole one. *)
  let encode key value =
    Printf.sprintf "%d %d\n%s%s" (String.length key) (String.length value) key
      value

  let decode path key contents =
    match
      Scanf.sscanf_opt contents "%u %u\n%n" (fun klen vlen start ->
          (klen, vlen, start))
    with
    | Some (klen, vlen, start) when start + klen + vlen = String.length contents
      ->
        if String.sub contents start klen <> key then None
        else Some (String.sub contents (start + klen) vlen)
    | _ -> failwith (path ^ ": malformed cache entry")

  let get ~table key =
    if not (enabled ()) then None
    else
      let path = entry_path table key in
      match In_channel.with_open_bin path In_channel.input_all with
      | contents -> decode path key contents
      | exception (Sys_error _ as e) ->
          if Sys.file_exists path then raise e else None

  let rec mkdir_p dir =
    if not (Sys.file_exists dir) then begin
      mkdir_p (Filename.dirname dir);
      try Sys.mkdir dir 0o755 with Sys_error _ when Sys.file_exists dir -> ()
    end

  (* The entry is written aside and renamed into place, so readers never see it
     half written. On Windows the rename fails while another process holds the
     entry open; that entry is complete, so it stays. *)
  let put ~table key value =
    if enabled () then begin
      let path = entry_path table key in
      let dir = Filename.dirname path in
      mkdir_p dir;
      let tmp, oc =
        Filename.open_temp_file ~mode:[ Open_binary ] ~temp_dir:dir
          (Filename.basename path ^ ".")
          ".tmp"
      in
      match
        output_string oc (encode key value);
        close_out oc
      with
      | exception e ->
          close_out_noerr oc;
          (try Sys.remove tmp with Sys_error _ -> ());
          raise e
      | () -> (
          try Sys.rename tmp path
          with Sys_error _ when Sys.file_exists path -> Sys.remove tmp)
    end

  (* Only files named like entries are removed, so a [CACHEDB] pointing at a
     directory holding other files keeps them. Entries and tables may vanish
     meanwhile, cleared by another process. *)
  let clear () =
    let remove file =
      try Sys.remove file
      with Sys_error _ when not (Sys.file_exists file) -> ()
    in
    let clear_table dir =
      Array.iter
        (fun f -> if is_entry f then remove (Filename.concat dir f))
        (Sys.readdir dir);
      if Sys.readdir dir = [||] then
        try Sys.rmdir dir
        with Sys_error _ when not (Sys.file_exists dir) -> ()
    in
    if Sys.file_exists cachedb then
      Array.iter
        (fun name ->
          let dir = Filename.concat cachedb name in
          if is_table_dir name && Sys.is_directory dir then clear_table dir)
        (Sys.readdir cachedb)
end
