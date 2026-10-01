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
    | [ n ] -> Type.pp_quoted ppf n
    | [ n; last ] ->
        Format.fprintf ppf "%a %s %a" Type.pp_quoted n conj Type.pp_quoted last
    | n :: ns -> Format.fprintf ppf "%a, %a" Type.pp_quoted n loop ns
  in
  loop ppf names

let missing name schema =
  let names = Schema.names schema in
  let nearest (d, ns) n =
    let dn = distance name n in
    if dn < d then (dn, [ n ]) else if dn = d then (d, n :: ns) else (d, ns)
  in
  match (names, List.fold_left nearest (2, []) names) with
  | [], _ -> v "no column %a. There are no columns." Type.pp_quoted name
  | _, (_, []) ->
      v "no column %a. The columns are %a." Type.pp_quoted name (pp_names "and")
        names
  | _, (_, near) ->
      v "no column %a. Did you mean %a?" Type.pp_quoted name (pp_names "or")
        (List.rev near)

let repeated ns =
  let add (seen, dups) n =
    if List.mem n seen && not (List.mem n dups) then (seen, n :: dups)
    else (n :: seen, dups)
  in
  List.rev (snd (List.fold_left add ([], []) ns))

let pp = Format.pp_print_string
