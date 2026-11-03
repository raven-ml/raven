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
   condition number times its forward function's bound; an inverse's [k (1 +
   |log (m / κ)|)] ulps, the error of the density [m / κ] its logarithm gives,
   plus [κ] times its forward function's bound, [κ] the scale; and a relative
   bound at float64 and at float32, alone or times the scale. *)
type bound =
  | Ulps of int
  | Near_zeros of int * int
  | Scaled of int
  | Inverse of int * int
  | Inverse_density of { k : int; m : float; forward : int }
  | Relative of float * float
  | Relative_scaled of float * float

let pp_bound ppf = function
  | Ulps k -> Format.fprintf ppf "%d ulps" k
  | Near_zeros (k, a) ->
      Format.fprintf ppf "%d ulps, or %d eps where below 1" k a
  | Scaled k -> Format.fprintf ppf "%d ulps, or %d eps times the scale" k k
  | Inverse (k, f) -> Format.fprintf ppf "%d + kappa * %d ulps" k f
  | Inverse_density { k; m; forward } ->
      Format.fprintf ppf "%d (1 + |log (%g / kappa)|) + kappa * %d ulps" k m
        forward
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

(* [Inverse_density]'s bound in ulps at the condition number [kappa]. *)
let density_ulps bound kappa =
  match bound with
  | Inverse_density { k; m; forward } ->
      (Float.of_int k *. (1. +. Float.abs (Float.log (m /. kappa))))
      +. (kappa *. Float.of_int forward)
  | _ -> invalid_arg "density_ulps"

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
    | Inverse_density _ -> density_ulps bound row.scale
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
    | Inverse_density _ -> d <= density_ulps bound row.scale
    | Relative (r64, r32) ->
        d = 0. || error <= (if f32 then r32 else r64) *. Float.abs r
    | Relative_scaled (r64, r32) ->
        d = 0.
        || error <= (if f32 then r32 else r64) *. row.scale *. Float.abs r

type f = {
  f : 'b. (float, 'b, Nx.host) Nx.t array -> (float, 'b, Nx.host) Nx.t;
}

(* [args dt rows] is each argument of [rows] as an array of [dt]. *)
let args (type b) (dt : (float, b) Nx.dtype) rows =
  let n = List.length rows in
  let arity = match rows with [] -> 0 | r :: _ -> Array.length r.args in
  Array.init arity (fun i ->
      Nx.cast dt
        (Nx.create Nx.float64 [| n |]
           (Array.of_list (List.map (fun r -> r.args.(i)) rows))))

(* [read y] is [y]'s elements as floats. *)
let read y = Nx.to_array (Nx.cast Nx.float64 y)

let eval (type b) (dt : (float, b) Nx.dtype) { f } rows =
  read (f (args dt rows))

(* [check_values ~bound file got] holds [got ~f32 rows], the values at the rows
   of [file] at one dtype, to [bound args] at every row, at float32 and float64,
   and fails listing the rows that miss it, worst first. *)
let check_values ~bound file got =
  List.map
    (fun (short, f32) ->
      test short (fun () ->
          let rows = rows file short in
          let got = got ~f32 rows in
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

(* [check ~bound file f] is [check_values] of [f] at each dtype. *)
let check ~bound file f =
  check_values ~bound file (fun ~f32 rows ->
      if f32 then eval Nx.float32 f rows else eval Nx.float64 f rows)

(* Functions *)

type u = { u : 'b. (float, 'b, Nx.host) Nx.t -> (float, 'b, Nx.host) Nx.t }

type b = {
  b :
    'b.
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t;
}

type c = {
  c :
    'b.
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t ->
    (float, 'b, Nx.host) Nx.t;
}

let everywhere b _ = b

let by_sign ~positive ~negative args =
  if args.(0) < 0. then negative else positive

(* [log_ulps k f] is [2k (1 + |log f|) + 4], an inverse's forward bound. *)
let log_ulps k f =
  (2. *. Float.of_int k *. (1. +. Float.abs (Stdlib.log (Float.abs f)))) +. 4.

(* Derivatives

   Each derivative is held to golden/special_grad/ (written by
   gen/special_grad.py) eagerly and compiled, at the budgets rune's derivatives
   of special functions promise: first derivatives within 16 times the
   function's bound, second derivatives within [2^-40] relative at float64 and
   [2^-16] at float32. *)

let grad_golden name = "golden/special_grad/" ^ name ^ ".golden"

(* Partials

   A function's partials share most of their program, and an incomplete beta's
   or gamma's takes seconds to compile, most of it Clang's. So the gradient in
   every argument is compiled once per dtype and run once, at the points of
   every partial's golden together, and each partial reads its own values.
   Functions that nx computes from one tail compile together, and so does a
   quantile's gradient with its second derivative: each pair's program holds
   that tail once. Eagerly each partial is taken alone.

   A program compiles once in each process that reads it, so a suite that
   reads one runs in one worker: in more, each would compile it again. *)

type g = {
  g : 'b. (float, 'b, Nx.host) Nx.t array -> (float, 'b, Nx.host) Nx.t array;
}

(* A compiled gradient's partials, by name: [golden name ~f32 rows] is the
   partial [name] at the rows of its golden, at the dtype [~f32] names, and
   [edge name ~f32] at the extra arguments the gradient ran at. *)
type joint = {
  golden : string -> f32:bool -> row list -> float array;
  edge : string -> f32:bool -> float array;
}

(* [joint ~edge names g] runs [g] once per dtype, at the rows of every golden of
   [names] and then at the arguments [edge]; the [i]th array [g] returns is the
   partial named by the [i]th of [names]. *)
let joint ?(edge = [||]) names { g } =
  let at (type b) (dt : (float, b) Nx.dtype) short =
    lazy
      (let rows = List.map (fun name -> rows (grad_golden name) short) names in
       let edge =
         Array.to_list
           (Array.map
              (fun args ->
                { args; value = Float.nan; overflow = false; scale = Float.nan })
              edge)
       in
       let got = Array.map read (g (args dt (List.concat (rows @ [ edge ])))) in
       let (_, golden), parts =
         List.fold_left_map
           (fun (i, start) r ->
             let n = List.length r in
             ((i + 1, start + n), Array.sub got.(i) start n))
           (0, 0) rows
       in
       let at_edge i = Array.sub got.(i) golden (List.length edge) in
       List.mapi
         (fun i (name, part) -> (name, (part, at_edge i)))
         (List.combine names parts))
  in
  let f32 = at Nx.float32 "f32" and f64 = at Nx.float64 "f64" in
  let get name single =
    List.assoc name (Lazy.force (if single then f32 else f64))
  in
  {
    golden = (fun name ~f32 _ -> fst (get name f32));
    edge = (fun name ~f32 -> snd (get name f32));
  }

(* [partial name ~bound ~eager compiled] is the partial [name] held eagerly and
   as [compiled] reads it, at both dtypes. *)
let partial name ~bound ~eager compiled =
  group name
    [
      group "eagerly" (check ~bound (grad_golden name) eager);
      group "compiled" (check_values ~bound (grad_golden name) compiled);
    ]

(* In [a], [2^-40 (1 + |log f|)] relative at float64 and [2^-16 (1 + |log f|)]
   at float32, the scale holding [1 + |log f|]. *)
let in_a _ = Relative_scaled (0x1p-40, 0x1p-16)
