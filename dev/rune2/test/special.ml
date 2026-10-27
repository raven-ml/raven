(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Goldens of derivatives of special functions, and their bounds.

   A golden lists, per dtype, the arguments of a function and its value
   correctly rounded to the dtype, as hexadecimal floats, and for some a last
   column a bound reads: an inverse's condition number, or the scale of an
   absolute bound. *)

open Windtrap

(* [overflow] when [value] is an infinity a finite exact value rounds to past
   the dtype's range. *)
type row = { args : float array; value : float; overflow : bool; scale : float }

(* The rows of [file] at the dtype named [short] ("f32", "f64"). *)
let rows file short =
  let path = Filename.concat (Filename.dirname Sys.executable_name) file in
  let lines = In_channel.with_open_bin path In_channel.input_lines in
  match List.filter (fun l -> not (String.starts_with ~prefix:"#" l)) lines with
  | [] -> failwith (file ^ ": no header")
  | header :: lines ->
      (* The columns are the dtype, the arguments, [f] and maybe a scale. *)
      let columns = String.split_on_char '\t' header in
      let scaled = List.nth columns (List.length columns - 1) <> "f" in
      let n = List.length columns - if scaled then 3 else 2 in
      let float = function
        | "overflow" -> Float.infinity
        | "-overflow" -> Float.neg_infinity
        | cell -> float_of_string cell
      in
      List.filter_map
        (fun line ->
          match String.split_on_char '\t' line with
          | dtype :: cells when dtype = short ->
              let overflow =
                match List.nth cells n with
                | "overflow" | "-overflow" -> true
                | _ -> false
              in
              let cells = Array.of_list (List.map float cells) in
              Some
                {
                  args = Array.sub cells 0 n;
                  value = cells.(n);
                  overflow;
                  scale = (if scaled then cells.(n + 1) else Float.nan);
                }
          | _ -> None)
        lines

(* The forms of a bound: [k] ulps; [k] ulps or [|f̂ - f| <= a eps] where [|f| <
   1]; [k] ulps or [|f̂ - f| <= k eps scale]; an inverse's [k] ulps plus its
   condition number times its forward function's bound; and a relative bound at
   float64 and at float32, alone or times the scale. *)
type bound =
  | Ulps of int
  | Near_zeros of int * int
  | Scaled of int
  | Inverse of int * int
  | Relative of float * float
  | Relative_scaled of float * float

let pp_bound ppf = function
  | Ulps k -> Format.fprintf ppf "%d ulps" k
  | Near_zeros (k, a) ->
      Format.fprintf ppf "%d ulps, or %d eps where below 1" k a
  | Scaled k -> Format.fprintf ppf "%d ulps, or %d eps times the scale" k k
  | Inverse (k, f) -> Format.fprintf ppf "%d + kappa * %d ulps" k f
  | Relative (r64, r32) ->
      Format.fprintf ppf "%g relative (%g at float32)" r64 r32
  | Relative_scaled (r64, r32) ->
      Format.fprintf ppf "%g relative (%g at float32) times the scale" r64 r32

(* The position of a float of [width] bits [b] among its format's values in
   order, [+0] at 0 and [-0] at -1. *)
let rank ~width b =
  let sign = Int64.shift_left 1L (width - 1) in
  if Int64.logand b sign = 0L then b
  else Int64.(sub (neg (logand b (sub sign 1L))) 1L)

let rank_of ~f32 x =
  if f32 then
    rank ~width:32
      (Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xffff_ffffL)
  else rank ~width:64 (Int64.bits_of_float x)

(* The distance between [a] and [b] in floats, as a float: zero between two
   NaNs, infinite between a NaN and a number. *)
let ulps ~f32 a b =
  if Float.is_nan a || Float.is_nan b then
    if Float.is_nan a && Float.is_nan b then 0. else Float.infinity
  else Int64.to_float (Int64.abs (Int64.sub (rank_of ~f32 a) (rank_of ~f32 b)))

(* Whether [got] meets [bound] against the correctly rounded [row]: exactly
   where the value is an exact infinity or NaN. A derivative's zero is held
   without its sign. An overflow, read as the infinity one rank past the largest
   float, is held by rank to the bound in ulps. *)
let within ~f32 bound row got =
  let eps = if f32 then 0x1p-23 else 0x1p-52 in
  let r = row.value in
  let d = ulps ~f32 r got in
  let error = Float.abs (got -. r) in
  if (not (Float.is_finite r)) && not row.overflow then d = 0.
  else if r = 0. && got = 0. then true
  else if row.overflow then
    d
    <=
    match bound with
    | Ulps k | Near_zeros (k, _) | Scaled k -> Float.of_int k
    | Inverse (k, forward) ->
        Float.of_int k +. (row.scale *. Float.of_int forward)
    | Relative (r64, r32) -> (if f32 then r32 else r64) /. eps
    | Relative_scaled (r64, r32) ->
        (if f32 then r32 else r64) *. row.scale /. eps
  else
    match bound with
    | Ulps k -> d <= Float.of_int k
    | Near_zeros (k, a) ->
        d <= Float.of_int k
        || (Float.abs r < 1. && error <= Float.of_int a *. eps)
    | Scaled k ->
        d <= Float.of_int k || error <= Float.of_int k *. eps *. row.scale
    | Inverse (k, forward) ->
        d <= Float.of_int k +. (row.scale *. Float.of_int forward)
    | Relative (r64, r32) ->
        d = 0. || error <= (if f32 then r32 else r64) *. Float.abs r
    | Relative_scaled (r64, r32) ->
        d = 0.
        || error <= (if f32 then r32 else r64) *. row.scale *. Float.abs r

type f = {
  f : 'b. (float, 'b, Nx.host) Nx.t array -> (float, 'b, Nx.host) Nx.t;
}

let eval (type b) (dt : (float, b) Nx.dtype) { f } rows =
  let n = List.length rows in
  let arity = match rows with [] -> 0 | r :: _ -> Array.length r.args in
  let args =
    Array.init arity (fun i ->
        Nx.cast dt
          (Nx.create Nx.float64 [| n |]
             (Array.of_list (List.map (fun r -> r.args.(i)) rows))))
  in
  Nx.to_array (Nx.cast Nx.float64 (f args))

(* [check ~bound file f] holds [f] to [bound args] at every row of [file], at
   float32 and float64, and fails listing the rows that miss it, worst first. *)
let check ~bound file f =
  List.map
    (fun (short, f32) ->
      test short (fun () ->
          let rows = rows file short in
          let got =
            if f32 then eval Nx.float32 f rows else eval Nx.float64 f rows
          in
          let misses =
            List.concat
              (List.mapi
                 (fun i row ->
                   let b = bound row.args in
                   if within ~f32 b row got.(i) then []
                   else [ (ulps ~f32 row.value got.(i), row, got.(i), b) ])
                 rows)
          in
          let misses =
            List.sort (fun (a, _, _, _) (b, _, _, _) -> compare b a) misses
          in
          match misses with
          | [] -> ()
          | _ ->
              let pp_miss ppf (d, row, got, b) =
                Format.fprintf ppf "at (%a): %h, not %h (%g ulps), bound %a"
                  (Format.pp_print_list
                     ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
                     (fun ppf x -> Format.fprintf ppf "%h" x))
                  (Array.to_list row.args) got row.value d pp_bound b;
                if Float.is_finite row.scale then
                  Format.fprintf ppf " (scale %g)" row.scale
              in
              failf "%d of %d rows miss their bound:@\n%a" (List.length misses)
                (List.length rows)
                (Format.pp_print_list pp_miss)
                (List.filteri (fun i _ -> i < 8) misses)))
    [ ("f32", true); ("f64", false) ]
