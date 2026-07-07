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

  let int ~key ~default =
    let value = getenv key default in
    { key; value = ref value }

  let string ~key ~default =
    let value =
      match Sys.getenv_opt key with
      | Some s ->
          let v = String.trim s in
          if v = "" then default else v
      | None -> default
    in
    { key; value = ref value }

  let get v = !(v.value)

  type binding = B : 'a t * 'a -> binding

  let with_context overrides f =
    let saved = List.map (fun (B (v, _)) -> B (v, !(v.value))) overrides in
    List.iter (fun (B (v, x)) -> v.value := x) overrides;
    Fun.protect
      ~finally:(fun () -> List.iter (fun (B (v, old)) -> v.value := old) saved)
      f
end

(* Whether training-mode behaviour (e.g. dropout) is active. *)
let training = Context_var.int ~key:"TRAINING" ~default:0

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

(* SHA-256 of a message short enough to fit a single 64-byte block after
   padding (at most 55 bytes). Returns the 32-byte digest. *)
let sha256 msg =
  let len = Bytes.length msg in
  if len > 55 then invalid_arg "Helpers.sha256: message exceeds one block";
  let block = Bytes.make 64 '\000' in
  Bytes.blit msg 0 block 0 len;
  Bytes.set block len '\x80';
  Bytes.set_int64_be block 56 (Int64.of_int (len * 8));
  let mask = 0xFFFFFFFF in
  let rotr x n = ((x lsr n) lor (x lsl (32 - n))) land mask in
  let w = Array.make 64 0 in
  for i = 0 to 15 do
    w.(i) <- Int32.to_int (Bytes.get_int32_be block (i * 4)) land mask
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
  let h = [| 0x6a09e667; 0xbb67ae85; 0x3c6ef372; 0xa54ff53a;
             0x510e527f; 0x9b05688c; 0x1f83d9ab; 0x5be0cd19 |] in
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
  let digest = Bytes.create 32 in
  Array.iteri
    (fun i x ->
      Bytes.set_int32_be digest (i * 4) (Int32.of_int ((h.(i) + x) land mask)))
    [| !a; !b; !c; !d; !e; !f; !g; !hh |];
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
