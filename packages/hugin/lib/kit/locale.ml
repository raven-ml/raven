(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  decimal : string;
  group : string;
  grouping : int list;
  minus : string;
  months : string array;
}

let english_months =
  [|
    "Jan";
    "Feb";
    "Mar";
    "Apr";
    "May";
    "Jun";
    "Jul";
    "Aug";
    "Sep";
    "Oct";
    "Nov";
    "Dec";
  |]

let check_text what s =
  if not (String.is_valid_utf_8 s) then
    invalid_arg (Printf.sprintf "Locale.v: %s is not valid UTF-8" what)

let check_nonempty what s =
  if s = "" then invalid_arg (Printf.sprintf "Locale.v: empty %s" what);
  check_text what s

let v ?(decimal = ".") ?(group = ",") ?(grouping = [ 3 ]) ?(minus = "\u{2212}")
    ?(months = english_months) () =
  check_nonempty "decimal separator" decimal;
  check_text "group separator" group;
  check_nonempty "minus sign" minus;
  if group <> "" && group = decimal then
    invalid_arg "Locale.v: the group separator is the decimal separator";
  if grouping = [] then invalid_arg "Locale.v: empty grouping";
  if List.exists (fun n -> n < 1) grouping then
    invalid_arg "Locale.v: a group size is below 1";
  if Array.length months <> 12 then
    invalid_arg "Locale.v: months does not have twelve names";
  Array.iter (check_nonempty "month name") months;
  (* The last size repeats, so repeats of it that end the list change
     nothing. *)
  let rec drop_repeats = function
    | n :: (n' :: _ as rest) when n = n' -> drop_repeats rest
    | l -> l
  in
  let grouping = List.rev (drop_repeats (List.rev grouping)) in
  { decimal; group; grouping; minus; months = Array.copy months }

let default = v ()
let decimal l = l.decimal
let group l = l.group
let grouping l = l.grouping
let minus l = l.minus

let month l m =
  if m < 1 || m > 12 then invalid_arg "Locale.month: month not in [1;12]";
  l.months.(m - 1)

let equal l l' =
  String.equal l.decimal l'.decimal
  && String.equal l.group l'.group
  && List.equal Int.equal l.grouping l'.grouping
  && String.equal l.minus l'.minus
  && Array.for_all2 String.equal l.months l'.months

let pp ppf l =
  let str ppf s = Format.fprintf ppf "\"%s\"" s in
  Format.fprintf ppf
    "@[<1>(locale@ (decimal %a)@ (group %a)@ (grouping (%a))@ (minus %a)@ \
     (months (%a)))@]"
    str l.decimal str l.group
    (Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_int)
    l.grouping str l.minus
    (Format.pp_print_array ~pp_sep:Format.pp_print_space str)
    l.months
