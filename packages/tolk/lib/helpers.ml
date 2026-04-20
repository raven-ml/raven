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

let amx = getenv "AMX" 0 <> 0
let allow_half8 = getenv "ALLOW_HALF8" 0 <> 0

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
