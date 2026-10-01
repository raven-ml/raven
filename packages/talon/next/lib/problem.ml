(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = string

let v fmt = Format.kasprintf Fun.id fmt

(* [distance a b] is the Levenshtein distance between [a] and [b], in bytes. *)
let distance a b =
  let m = String.length a and n = String.length b in
  let row = Array.init (n + 1) Fun.id in
  for i = 1 to m do
    let diag = ref row.(0) in
    row.(0) <- i;
    for j = 1 to n do
      let cost = if Char.equal a.[i - 1] b.[j - 1] then 0 else 1 in
      let next = min (min (row.(j) + 1) (row.(j - 1) + 1)) (!diag + cost) in
      diag := row.(j);
      row.(j) <- next
    done
  done;
  row.(n)

let pp_names conj ppf names =
  let rec loop ppf = function
    | [] -> ()
    | [ n ] -> Format.fprintf ppf "%S" n
    | [ n; last ] -> Format.fprintf ppf "%S %s %S" n conj last
    | n :: ns -> Format.fprintf ppf "%S, %a" n loop ns
  in
  loop ppf names

let missing name schema =
  let names = Schema.names schema in
  let close =
    List.filter_map
      (fun n ->
        let d = distance name n in
        if d <= 2 then Some (d, n) else None)
      names
  in
  match
    (List.stable_sort (fun (d0, _) (d1, _) -> Int.compare d0 d1) close, names)
  with
  | [], [] -> v "no column %S. There are no columns." name
  | [], names ->
      v "no column %S. The columns are %a." name (pp_names "and") names
  | close, _ ->
      v "no column %S. Did you mean %a?" name (pp_names "or")
        (List.map snd close)

let repeated ns =
  let add (seen, dups) n =
    if List.mem n seen && not (List.mem n dups) then (seen, n :: dups)
    else (n :: seen, dups)
  in
  List.rev (snd (List.fold_left add ([], []) ns))

let pp = Format.pp_print_string
