(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Environment *)

let getenv name default =
  match Sys.getenv_opt name with
  | Some s -> (try int_of_string s with Failure _ -> default)
  | None -> default

let getenv_str name default =
  match Sys.getenv_opt name with
  | Some s when s <> "" -> s
  | _ -> default

let select_first_inited ~message candidates =
  let rec select errors = function
    | [] ->
        (match errors with
         | [ exn, backtrace ] -> Printexc.raise_with_backtrace exn backtrace
         | _ ->
             let reasons = List.rev_map (fun (exn, _) -> Printexc.to_string exn) errors in
             failwith (String.concat "\n" (message :: reasons)))
    | create :: rest ->
        match create () with
        | value -> value
        | exception (Out_of_memory | Stack_overflow | Sys.Break as exn) -> raise exn
        | exception exn ->
            let backtrace = Printexc.get_raw_backtrace () in
            select ((exn, backtrace) :: errors) rest
  in
  select [] candidates

let allow_half8 = getenv "ALLOW_HALF8" 0 <> 0

(* Canonical device name: uppercase the backend part and strip a ":0"
   suffix, e.g. "cpu:0" -> "CPU". *)
let canonicalize_device_name device =
  let device =
    match String.index_opt device ':' with
    | Some i ->
        String.uppercase_ascii (String.sub device 0 i)
        ^ String.sub device i (String.length device - i)
    | None -> String.uppercase_ascii device
  in
  let len = String.length device in
  if len >= 2 && String.equal (String.sub device (len - 2) 2) ":0" then
    String.sub device 0 (len - 2)
  else device

(* Context variables *)

module Context_var = struct
  type 'a t = { key : string; value : 'a ref }

  let declared : (string, unit) Hashtbl.t = Hashtbl.create 64

  let declare key =
    if Hashtbl.mem declared key then
      invalid_arg (Printf.sprintf "Context_var: %s is already declared" key);
    Hashtbl.replace declared key ()

  let make ~key ~default ~parse =
    declare key;
    { key; value = ref (match Sys.getenv_opt key with
        | None -> default
        | Some s -> parse s) }

  let int ~key ~default =
    make ~key ~default
      ~parse:(fun s -> try int_of_string s with Failure _ -> default)

  let string ~key ~default =
    make ~key ~default ~parse:(fun s ->
        let s = String.trim s in
        if s = "" then default else s)

  let key v = v.key
  let get v = !(v.value)

  type binding = B : 'a t * 'a -> binding

  let with_context overrides f =
    let saved = List.map (fun (B (v, _)) -> B (v, !(v.value))) overrides in
    List.iter (fun (B (v, x)) -> v.value := x) overrides;
    Fun.protect
      ~finally:(fun () -> List.iter (fun (B (v, old)) -> v.value := old) saved)
      f
end

let dev =
  Context_var.make ~key:"DEV" ~default:[ Tolk_uop.Target.of_string "" ]
    ~parse:(fun s -> List.map Tolk_uop.Target.of_string (String.split_on_char ';' s))

let target ?(arch = "") device =
  let open Tolk_uop.Target in
  let device = String.uppercase_ascii (List.hd (String.split_on_char ':' device)) in
  let targets = Context_var.get dev in
  let t = match List.find_opt (fun t -> t.device = "" || t.device = device) targets with
    | Some t -> t
    | None -> of_string device
  in
  let key = device ^ "_CC" in
  let old = getenv_str key "" in
  if old <> "" then
    invalid_arg (Printf.sprintf "%s=%s is deprecated, use DEV=%s instead"
      key old (to_string { t with device; renderer = old }));
  { t with device; arch = (if t.arch = "" then arch else t.arch) }

let select_interface ~device candidates =
  let t = target device in
  let key = t.device ^ "_IFACE" in
  let old = getenv_str key "" in
  if old <> "" then
    invalid_arg (Printf.sprintf "%s=%s is deprecated, use DEV=%s instead"
      key old (Tolk_uop.Target.to_string { t with interface = old }));
  let candidates = List.filter (fun (name, _) ->
      (t.interface = "" || name = t.interface)
      && (String.starts_with ~prefix:"MOCK" t.interface
          || not (String.starts_with ~prefix:"MOCK" name))) candidates in
  if candidates = [] then
    invalid_arg (Printf.sprintf "%s has no interface %S" t.device t.interface);
  select_first_inited ~message:(Printf.sprintf "No interface for %s is available" device)
    (List.map snd candidates)

(* Each variable is declared once, here, so every reader shares one value and
   a [with_context] override reaches all of them. *)

let noopt = Context_var.int ~key:"NOOPT" ~default:0
let image = Context_var.int ~key:"IMAGE" ~default:0
let float16 = Context_var.int ~key:"FLOAT16" ~default:0
let openpilot_hacks = Context_var.int ~key:"OPENPILOT_HACKS" ~default:0

(* Whether training-mode behaviour (e.g. dropout) is active. *)
let training = Context_var.int ~key:"TRAINING" ~default:0
let use_tc = Context_var.int ~key:"TC" ~default:1
let tc_select = Context_var.int ~key:"TC_SELECT" ~default:(-1)
let tc_opt = Context_var.int ~key:"TC_OPT" ~default:0
let transcendental = Context_var.int ~key:"TRANSCENDENTAL" ~default:1
let nolocals = Context_var.int ~key:"NOLOCALS" ~default:0
let split_reduceop = Context_var.int ~key:"SPLIT_REDUCEOP" ~default:1

let reduceop_split_threshold =
  Context_var.int ~key:"REDUCEOP_SPLIT_THRESHOLD" ~default:32768

let reduceop_split_size = Context_var.int ~key:"REDUCEOP_SPLIT_SIZE" ~default:22
let lru = Context_var.int ~key:"LRU" ~default:1
let ring = Context_var.int ~key:"RING" ~default:1
let all2all = Context_var.int ~key:"ALL2ALL" ~default:0

let ring_allreduce_threshold =
  Context_var.int ~key:"RING_ALLREDUCE_THRESHOLD" ~default:256_000

let allreduce_cast = Context_var.int ~key:"ALLREDUCE_CAST" ~default:1
let disable_fast_idiv = Context_var.int ~key:"DISABLE_FAST_IDIV" ~default:1
let max_kernel_buffers = Context_var.int ~key:"MAX_KERNEL_BUFFERS" ~default:0

(* Partial contiguous in rangeify. *)
let pcontig = Context_var.int ~key:"PCONTIG" ~default:0

(* Allow TF32 on NVIDIA GPUs. *)
let allow_tf32 = Context_var.int ~key:"ALLOW_TF32" ~default:0

(* Terminal output *)

let no_color = Context_var.int ~key:"NO_COLOR" ~default:0

let colors =
  [ "black"; "red"; "green"; "yellow"; "blue"; "magenta"; "cyan"; "white" ]

(* An uppercase [color] is the bright variant. *)
let colored ?(background = false) st color =
  match color with
  | None -> st
  | Some _ when Context_var.get no_color <> 0 -> st
  | Some color ->
      let index =
        let lower = String.lowercase_ascii color in
        let rec find i = function
          | [] -> invalid_arg ("colored: unknown color " ^ color)
          | c :: rest -> if String.equal c lower then i else find (i + 1) rest
        in
        find 0 colors
      in
      let bright = String.equal (String.uppercase_ascii color) color in
      Printf.sprintf "\027[%dm%s\027[0m"
        ((if background then 10 else 0) + (if bright then 60 else 0) + 30 + index)
        st

(* Drops the escape sequences ESC [ K and ESC [ ... m. *)
let ansistrip s =
  let n = String.length s in
  let b = Buffer.create n in
  let rec go i =
    if i < n then
      if s.[i] = '\027' && i + 1 < n && s.[i + 1] = '[' then
        if i + 2 < n && s.[i + 2] = 'K' then go (i + 3)
        else
          match String.index_from_opt s (i + 2) 'm' with
          | Some m -> go (m + 1)
          | None ->
              Buffer.add_char b s.[i];
              go (i + 1)
      else begin
        Buffer.add_char b s.[i];
        go (i + 1)
      end
  in
  go 0;
  Buffer.contents b

let ansilen s = String.length (ansistrip s)

let time_to_str ?(w = 8) t =
  if t > 10.0 then Printf.sprintf "%*.2fs " w t
  else if t > 10.0 /. 1e3 then Printf.sprintf "%*.2fms" w (t *. 1e3)
  else Printf.sprintf "%*.2fus" w (t *. 1e6)

let size_to_str s =
  let f = float_of_int s in
  if s >= 1 lsl 30 then Printf.sprintf "%.2f GB" (f /. float_of_int (1 lsl 30))
  else if s >= 1 lsl 20 then
    Printf.sprintf "%.2f MB" (f /. float_of_int (1 lsl 20))
  else if s >= 1 lsl 10 then
    Printf.sprintf "%.2f KB" (f /. float_of_int (1 lsl 10))
  else Printf.sprintf "%d B" s

(* [mem_used] and [mem_used_per_device] follow live allocations and are not
   reset. *)
module Global_counters = struct
  let global_ops = ref 0
  let global_mem = ref 0
  let time_sum_s = ref 0.0
  let kernel_count = ref 0
  let mem_used = ref 0
  let mem_used_per_device : (string, int) Hashtbl.t = Hashtbl.create 4

  let add_mem_used device nbytes =
    mem_used := !mem_used + nbytes;
    let prev =
      Option.value (Hashtbl.find_opt mem_used_per_device device) ~default:0
    in
    Hashtbl.replace mem_used_per_device device (prev + nbytes)

  let reset () =
    global_ops := 0;
    global_mem := 0;
    time_sum_s := 0.0;
    kernel_count := 0
end

(* Hashing *)

let sha256_k =
  [|
    0x428a2f98; 0x71374491; 0xb5c0fbcf; 0xe9b5dba5; 0x3956c25b; 0x59f111f1;
    0x923f82a4; 0xab1c5ed5; 0xd807aa98; 0x12835b01; 0x243185be; 0x550c7dc3;
    0x72be5d74; 0x80deb1fe; 0x9bdc06a7; 0xc19bf174; 0xe49b69c1; 0xefbe4786;
    0x0fc19dc6; 0x240ca1cc; 0x2de92c6f; 0x4a7484aa; 0x5cb0a9dc; 0x76f988da;
    0x983e5152; 0xa831c66d; 0xb00327c8; 0xbf597fc7; 0xc6e00bf3; 0xd5a79147;
    0x06ca6351; 0x14292967; 0x27b70a85; 0x2e1b2138; 0x4d2c6dfc; 0x53380d13;
    0x650a7354; 0x766a0abb; 0x81c2c92e; 0x92722c85; 0xa2bfe8a1; 0xa81a664b;
    0xc24b8b70; 0xc76c51a3; 0xd192e819; 0xd6990624; 0xf40e3585; 0x106aa070;
    0x19a4c116; 0x1e376c08; 0x2748774c; 0x34b0bcb5; 0x391c0cb3; 0x4ed8aa4a;
    0x5b9cca4f; 0x682e6ff3; 0x748f82ee; 0x78a5636f; 0x84c87814; 0x8cc70208;
    0x90befffa; 0xa4506ceb; 0xbef9a3f7; 0xc67178f2;
  |]

(* SHA-256 of a message of any length. Returns the 32-byte digest. *)
let sha256 msg =
  let len = Bytes.length msg in
  let mask = 0xFFFFFFFF in
  let rotr x n = ((x lsr n) lor (x lsl (32 - n))) land mask in
  let h = [| 0x6a09e667; 0xbb67ae85; 0x3c6ef372; 0xa54ff53a;
             0x510e527f; 0x9b05688c; 0x1f83d9ab; 0x5be0cd19 |] in
  let w = Array.make 64 0 in
  let compress block pos =
    for i = 0 to 15 do
      w.(i) <- Int32.to_int (Bytes.get_int32_be block (pos + (i * 4))) land mask
    done;
    for i = 16 to 63 do
      let s0 =
        rotr w.(i - 15) 7 lxor rotr w.(i - 15) 18 lxor (w.(i - 15) lsr 3)
      in
      let s1 =
        rotr w.(i - 2) 17 lxor rotr w.(i - 2) 19 lxor (w.(i - 2) lsr 10)
      in
      w.(i) <- (w.(i - 16) + s0 + w.(i - 7) + s1) land mask
    done;
    let a = ref h.(0) and b = ref h.(1) and c = ref h.(2) and d = ref h.(3) in
    let e = ref h.(4) and f = ref h.(5) and g = ref h.(6) and hh = ref h.(7) in
    for i = 0 to 63 do
      let s1 = rotr !e 6 lxor rotr !e 11 lxor rotr !e 25 in
      let ch = !e land !f lxor (lnot !e land !g) in
      let t1 = (!hh + s1 + ch + sha256_k.(i) + w.(i)) land mask in
      let s0 = rotr !a 2 lxor rotr !a 13 lxor rotr !a 22 in
      let maj = !a land !b lxor (!a land !c) lxor (!b land !c) in
      let t2 = (s0 + maj) land mask in
      hh := !g; g := !f; f := !e; e := (!d + t1) land mask;
      d := !c; c := !b; b := !a; a := (t1 + t2) land mask
    done;
    h.(0) <- (h.(0) + !a) land mask; h.(1) <- (h.(1) + !b) land mask;
    h.(2) <- (h.(2) + !c) land mask; h.(3) <- (h.(3) + !d) land mask;
    h.(4) <- (h.(4) + !e) land mask; h.(5) <- (h.(5) + !f) land mask;
    h.(6) <- (h.(6) + !g) land mask; h.(7) <- (h.(7) + !hh) land mask
  in
  for blk = 0 to (len / 64) - 1 do
    compress msg (blk * 64)
  done;
  let rem = len land 63 in
  let tail = Bytes.make (if rem < 56 then 64 else 128) '\000' in
  Bytes.blit msg (len - rem) tail 0 rem;
  Bytes.set tail rem '\x80';
  Bytes.set_int64_be tail (Bytes.length tail - 8) (Int64.of_int (len * 8));
  compress tail 0;
  if Bytes.length tail = 128 then compress tail 64;
  let digest = Bytes.create 32 in
  Array.iteri
    (fun i x -> Bytes.set_int32_be digest (i * 4) (Int32.of_int x))
    h;
  digest

(* Collections *)

(* Preserves first occurrence, removes duplicates. *)
let dedup_by eq lst =
  let rec loop acc = function
    | [] -> List.rev acc
    | x :: rest ->
        if List.exists (eq x) acc then loop acc rest
        else loop (x :: acc) rest
  in
  loop [] lst

(* Partitions old_shape indices into contiguous groups whose cumulative products
   match the corresponding new_shape elements, returning None if no valid
   partition exists. Used to determine whether a reshape is a simple view
   (contraction of contiguous axes) or requires a copy. *)
let get_contraction old_shape new_shape =
  let n_old = Array.length old_shape in
  let n_new = Array.length new_shape in
  let acc_old = Array.make n_old 1 in
  let acc_new = Array.make n_new 1 in
  if n_old > 0 then acc_old.(0) <- old_shape.(0);
  for i = 1 to n_old - 1 do
    acc_old.(i) <- acc_old.(i - 1) * old_shape.(i)
  done;
  if n_new > 0 then acc_new.(0) <- new_shape.(0);
  for i = 1 to n_new - 1 do
    acc_new.(i) <- acc_new.(i - 1) * new_shape.(i)
  done;
  let split = Array.make n_new 0 in
  let ok = ref true in
  for i = 0 to n_new - 1 do
    if !ok then begin
      if acc_new.(i) = 1 then split.(i) <- 0
      else
        match
          let found = ref (-1) in
          for j = 0 to n_old - 1 do
            if !found = -1 && acc_old.(j) = acc_new.(i) then found := j + 1
          done;
          !found
        with
        | -1 -> ok := false
        | idx -> split.(i) <- idx
    end
  done;
  if not !ok then None
  else
    let starts = Array.make n_new 0 in
    let ends = Array.make n_new 0 in
    for i = 0 to n_new - 1 do
      starts.(i) <- (if i = 0 then 0 else split.(i - 1));
      ends.(i) <- (if i = n_new - 1 then n_old else split.(i))
    done;
    Some
      (Array.to_list
         (Array.init n_new (fun i ->
              List.init (ends.(i) - starts.(i)) (fun j -> starts.(i) + j))))
