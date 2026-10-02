(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's planner: every kernel that combines elements writes the same bytes
   at every plan thread count, and the same as under nx.cpu's own policy. *)

open Windtrap
module View = Nx_array.View

type ('a, 'b) arr = ('a, 'b) Nx_array.t
type index = (int64, Nx_dtype.int64_elt) arr

(* The planner's entry points, with the plan's thread count last, under
   kernels.ml's C names. *)

external reduce_sum : ('a, 'b) arr -> ('a, 'b) arr -> int array -> int -> unit
  = "caml_nx_c_reduce_sum"

external reduce_prod : ('a, 'b) arr -> ('a, 'b) arr -> int array -> int -> unit
  = "caml_nx_c_reduce_prod"

external reduce_max : ('a, 'b) arr -> ('a, 'b) arr -> int array -> int -> unit
  = "caml_nx_c_reduce_max"

external reduce_min : ('a, 'b) arr -> ('a, 'b) arr -> int array -> int -> unit
  = "caml_nx_c_reduce_min"

external argmax : index -> ('a, 'b) arr -> int -> int -> unit
  = "caml_nx_c_argmax"

external argmin : index -> ('a, 'b) arr -> int -> int -> unit
  = "caml_nx_c_argmin"

external cumsum : ('a, 'b) arr -> ('a, 'b) arr -> int -> int -> unit
  = "caml_nx_c_cumsum"

external cumprod : ('a, 'b) arr -> ('a, 'b) arr -> int -> int -> unit
  = "caml_nx_c_cumprod"

external cummax : ('a, 'b) arr -> ('a, 'b) arr -> int -> int -> unit
  = "caml_nx_c_cummax"

external cummin : ('a, 'b) arr -> ('a, 'b) arr -> int -> int -> unit
  = "caml_nx_c_cummin"

external sort : ('a, 'b) arr -> ('a, 'b) arr -> int -> bool -> int -> unit
  = "caml_nx_c_sort"

external argsort : index -> ('a, 'b) arr -> int -> bool -> int -> unit
  = "caml_nx_c_argsort"

external group_rows : index -> (int64, Nx_dtype.uint64_elt) arr -> int -> unit
  = "caml_nx_c_group"

(* Thread counts. [None] is nx.cpu's policy, through its kernels; the last runs
   on every core. *)

module Cpu = (val Nx_backend.kernels Nx_cpu.backend)

let reference = Some 1

let counts =
  [ None; Some 2; Some 3; Some 8; Some (Domain.recommended_domain_count ()) ]

let label = function
  | None -> "nx.cpu's policy"
  | Some n -> Printf.sprintf "%d threads" n

let reduce (op : Nx_backend.reduce) threads ~axes x ~dst =
  match (threads, op) with
  | None, _ -> Cpu.reduce op ~axes x ~dst
  | Some n, Sum -> reduce_sum dst x axes n
  | Some n, Prod -> reduce_prod dst x axes n
  | Some n, Max -> reduce_max dst x axes n
  | Some n, Min -> reduce_min dst x axes n

let scan (op : Nx_backend.reduce) threads ~axis x ~dst =
  match (threads, op) with
  | None, _ -> Cpu.scan op ~axis x ~dst
  | Some n, Sum -> cumsum dst x axis n
  | Some n, Prod -> cumprod dst x axis n
  | Some n, Max -> cummax dst x axis n
  | Some n, Min -> cummin dst x axis n

let arg_reduce (op : Nx_backend.arg_reduce) threads ~axis x ~dst =
  match (threads, op) with
  | None, _ -> Cpu.arg_reduce op ~axis x ~dst
  | Some n, Argmax -> argmax dst x axis n
  | Some n, Argmin -> argmin dst x axis n

let sort_with ~arg ~descending threads ~axis x ~dst ~idx =
  match (threads, arg) with
  | None, false -> Cpu.sort ~descending ~axis x ~dst
  | None, true -> Cpu.argsort ~descending ~axis x ~dst:idx
  | Some n, false -> sort dst x axis descending n
  | Some n, true -> argsort idx x axis descending n

let group_with threads x ~dst =
  match threads with None -> Cpu.group x ~dst | Some n -> group_rows dst x n

(* Arrays *)

type dtype = D : ('a, 'b) Nx_dtype.t -> dtype

let op_name : Nx_backend.reduce -> string = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"

(* A destination of [shape], C-contiguous, its bytes a fixed pattern. *)
let fresh dtype shape =
  let buffer = Nx_array.Elements.create dtype (Array.fold_left ( * ) 1 shape) in
  Bigarray.Array1.fill (Nx_device.Buffer.bigarray Bigarray.char buffer) '\xa5';
  { Nx_array.dtype; view = View.create shape; buffer }

let bytes_of (a : (_, _) arr) =
  let b = Nx_device.Buffer.bigarray Bigarray.char a.buffer in
  String.init (Bigarray.Array1.dim b) (Bigarray.Array1.get b)

(* Values. Floats are uniform in [-1, 1), or near 1 for products, so that a long
   sum's or product's last bits depend on its association. With [specials],
   about one in 64 is a NaN of a distinct payload, a signed zero, an infinity or
   a subnormal. Integers include their extremes, so sums wrap. *)

let specials =
  [|
    Int64.float_of_bits 0x7ff8_0000_0000_0001L;
    Int64.float_of_bits 0xfff8_0000_0000_0002L;
    Int32.float_of_bits 0x7fc0_0003l;
    0.;
    -0.;
    infinity;
    neg_infinity;
    1e-310;
    1e-40;
  |]

let float_value st ~specials:with_specials ~prod =
  if with_specials && Random.State.int st 64 = 0 then
    specials.(Random.State.int st (Array.length specials))
  else if prod then 1. +. Random.State.float st 0.002 -. 0.001
  else Random.State.float st 2. -. 1.

let value : type a b.
    Random.State.t -> specials:bool -> prod:bool -> (a, b) Nx_dtype.t -> a =
 fun st ~specials ~prod dtype ->
  let float () = float_value st ~specials ~prod in
  let extreme = Random.State.int st 32 in
  match dtype with
  | Float16 -> float ()
  | Float32 -> float ()
  | Float64 -> float ()
  | BFloat16 -> float ()
  | Complex64 -> { Complex.re = float (); im = float () }
  | Int32 -> (
      match extreme with
      | 0 -> Int32.min_int
      | 1 -> Int32.max_int
      | _ -> Random.State.bits32 st)
  | Int64 -> (
      match extreme with
      | 0 -> Int64.min_int
      | 1 -> Int64.max_int
      | _ -> Random.State.bits64 st)
  | UInt8 -> Random.State.int st 256
  | Bool -> Random.State.bool st
  | _ -> invalid_arg "test_planner: no values for this dtype"

(* A layout: a C-contiguous base buffer of [base] elements and the view of it
   the kernel reads. *)
type layout = { name : string; base : int array; view : View.t -> View.t }

let contiguous base =
  {
    name = String.concat "x" (List.map string_of_int (Array.to_list base));
    base;
    view = Fun.id;
  }

let transposed base =
  let l = contiguous base in
  {
    l with
    name = "transposed " ^ l.name;
    view = (fun v -> View.permute v [| 1; 0 |]);
  }

let flipped l =
  {
    l with
    name = l.name ^ " flipped on axis 0";
    view =
      (fun v ->
        let v = l.view v in
        View.flip v (Array.init (View.ndim v) (fun i -> i = 0)));
  }

let operand st ~specials ~prod dtype l =
  let n = Array.fold_left ( * ) 1 l.base in
  let buffer = Nx_array.Elements.create dtype n in
  let set = Nx_array.Elements.set dtype buffer in
  for i = 0 to n - 1 do
    set i (value st ~specials ~prod dtype)
  done;
  { Nx_array.dtype; view = l.view (View.create l.base); buffer }

(* The law *)

let outcome run threads =
  match run threads with
  | bytes -> Ok bytes
  | exception e -> Error (Printexc.to_string e)

let difference one other =
  match (one, other) with
  | Ok a, Ok b when String.length a = String.length b ->
      let i = ref 0 in
      while a.[!i] = b.[!i] do
        incr i
      done;
      Printf.sprintf "byte %d of %d differs" !i (String.length a)
  | Ok _, Ok _ -> "the lengths differ"
  | Error e, Ok _ -> Printf.sprintf "one thread raised %s, this did not" e
  | Ok _, Error e -> Printf.sprintf "it raised %s, one thread did not" e
  | Error a, Error b -> Printf.sprintf "it raised %s, one thread %s" b a

let same_bytes what run =
  let one = outcome run reference in
  List.iter
    (fun threads ->
      let other = outcome run threads in
      if other <> one then
        failf "%s: at %s, %s" what (label threads) (difference one other))
    counts

(* Each case draws a seed and runs it without, then with, specials. *)
let case name law =
  prop ~count:2 name (Gen.int_range 0 0x3fff_ffff) (fun seed ->
      law (seed, false);
      law (seed, true))

(* Reductions *)

let kept shape axes =
  Array.of_list
    (List.filteri (fun d _ -> not (Array.mem d axes)) (Array.to_list shape))

let reduction_law op l axes (seed, specials) (D dtype) =
  let st = Random.State.make [| seed |] in
  let x = operand st ~specials ~prod:(op = Nx_backend.Prod) dtype l in
  let shape = kept (View.shape x.view) axes in
  let run threads =
    let dst = fresh dtype shape in
    reduce op threads ~axes x ~dst;
    bytes_of dst
  in
  same_bytes
    (Printf.sprintf "%s of %s %s" (op_name op) (Nx_dtype.to_string dtype) l.name)
    run

let floats = [ D Float32; D Float64; D Float16; D BFloat16 ]
let ints = [ D Int32; D Int64; D UInt8 ]
let arith = floats @ (D Complex64 :: ints)
let ordered = floats @ ints @ [ D Bool ]

let reduction_dtypes : Nx_backend.reduce -> dtype list = function
  | Sum | Prod -> arith
  | Max | Min -> ordered

let reduction_layouts =
  let paths =
    [
      (* per-output: blocks within a run *)
      (contiguous [| 7; 3077 |], [| 1 |]);
      (* per-output: one run, and two axes merged into one *)
      (contiguous [| 5000 |], [| 0 |]);
      (contiguous [| 70; 70 |], [| 0; 1 |]);
      (* per-output: a permuted layout, walked in memory order *)
      (transposed [| 40; 130 |], [| 0; 1 |]);
      (* per-output: runs that cannot merge, blocks across them *)
      (contiguous [| 9; 5; 300 |], [| 0; 2 |]);
      (* streaming: rows across blocks *)
      (contiguous [| 3073; 37 |], [| 0 |]);
      (* streaming over three panels of three tiles *)
      (contiguous [| 5; 3; 40000 |], [| 0 |]);
      (* streaming with merged rows *)
      (contiguous [| 2100; 3; 50 |], [| 0; 1 |]);
      (* many outputs, for the claims of a dynamic schedule *)
      (contiguous [| 100; 200 |], [| 1 |]);
      (* streaming: many units, each over several blocks *)
      (contiguous [| 2100; 32; 8 |], [| 0 |]);
      (* per-output: many outputs, each over several blocks *)
      (contiguous [| 48; 2100 |], [| 1 |]);
    ]
  in
  let others =
    [
      (* a broadcast reduced axis: a run of stride 0 *)
      ( {
          name = "1x50 expanded to 3000x50";
          base = [| 1; 50 |];
          view = (fun v -> View.expand v [| 3000; 50 |]);
        },
        [| 0 |] );
      (* axes of extent 1 with arbitrary strides *)
      ( {
          name = "1x2048x1 with strides 999, 1, -5";
          base = [| 2048 |];
          view =
            (fun _ -> View.create ~strides:[| 999; 1; -5 |] [| 1; 2048; 1 |]);
        },
        [| 0; 1; 2 |] );
      ( {
          name = "3x1x2000 with strides 2000, 7, 1";
          base = [| 6000 |];
          view =
            (fun _ -> View.create ~strides:[| 2000; 7; 1 |] [| 3; 1; 2000 |]);
        },
        [| 0; 1 |] );
      (* empty: sum stores the identity, max raises *)
      (contiguous [| 0; 5 |], [| 0 |]);
      (contiguous [| 5; 0 |], [| 1 |]);
    ]
  in
  paths @ List.map (fun (l, axes) -> (flipped l, axes)) paths @ others

let axes_name axes =
  "over " ^ String.concat ", " (List.map string_of_int (Array.to_list axes))

let reductions =
  group "reductions"
    (List.concat_map
       (fun op ->
         List.map
           (fun (l, axes) ->
             case
               (Printf.sprintf "%s of %s %s" (op_name op) l.name
                  (axes_name axes))
               (fun v ->
                 List.iter (reduction_law op l axes v) (reduction_dtypes op)))
           reduction_layouts)
       [ Nx_backend.Sum; Prod; Max; Min ])

(* Arg-reductions *)

let arg_reductions =
  let layouts =
    [
      (contiguous [| 64; 3000 |], 1);
      (contiguous [| 3000; 64 |], 0);
      ( {
          (contiguous [| 64; 3000 |]) with
          name = "64x3000 flipped on axis 1";
          view = (fun v -> View.flip v [| false; true |]);
        },
        1 );
    ]
  in
  let law op l axis (seed, specials) (D dtype) =
    let st = Random.State.make [| seed |] in
    let x = operand st ~specials ~prod:false dtype l in
    let shape = kept (View.shape x.view) [| axis |] in
    let run threads =
      let dst = fresh Nx_dtype.Int64 shape in
      arg_reduce op threads ~axis x ~dst;
      bytes_of dst
    in
    same_bytes (Printf.sprintf "%s %s" (Nx_dtype.to_string dtype) l.name) run
  in
  group "arg-reductions"
    (List.concat_map
       (fun (op, name) ->
         List.map
           (fun (l, axis) ->
             case (Printf.sprintf "%s of %s along %d" name l.name axis)
               (fun v -> List.iter (law op l axis v) [ D Float32; D Int32 ]))
           layouts)
       [ (Nx_backend.Argmax, "argmax"); (Argmin, "argmin") ])

(* Scans *)

let scans =
  let layouts =
    [
      (contiguous [| 12295 |], 0);
      (contiguous [| 6; 9000 |], 1);
      (contiguous [| 9000; 6 |], 0);
      ( {
          (contiguous [| 6; 9000 |]) with
          name = "6x9000 flipped on axis 1";
          view = (fun v -> View.flip v [| false; true |]);
        },
        1 );
      (contiguous [| 4; 4095 |], 1);
      (contiguous [| 4; 4096 |], 1);
      (contiguous [| 4; 8192 |], 1);
      (contiguous [| 32; 9000 |], 1);
    ]
  in
  let dtypes : Nx_backend.reduce -> dtype list = function
    | Sum | Prod ->
        [ D Float32; D Float64; D Float16; D Complex64; D Int64; D UInt8 ]
    | Max | Min -> [ D Float32; D Float64; D Float16; D Int64; D UInt8; D Bool ]
  in
  let law op l axis (seed, specials) (D dtype) =
    let st = Random.State.make [| seed |] in
    let x = operand st ~specials ~prod:(op = Nx_backend.Prod) dtype l in
    let run threads =
      let dst = fresh dtype (View.shape x.view) in
      scan op threads ~axis x ~dst;
      bytes_of dst
    in
    same_bytes (Printf.sprintf "%s %s" (Nx_dtype.to_string dtype) l.name) run
  in
  group "scans"
    (List.concat_map
       (fun op ->
         List.map
           (fun (l, axis) ->
             case
               (Printf.sprintf "cum%s of %s along %d" (op_name op) l.name axis)
               (fun v -> List.iter (law op l axis v) (dtypes op)))
           layouts)
       [ Nx_backend.Sum; Prod; Max; Min ])

(* Sorts *)

let sorts =
  let layouts =
    [ (contiguous [| 40; 300 |], 1); (contiguous [| 300; 40 |], 0) ]
  in
  let law ~arg ~descending l axis (seed, specials) (D dtype) =
    let st = Random.State.make [| seed |] in
    let x = operand st ~specials ~prod:false dtype l in
    let shape = View.shape x.view in
    let run threads =
      let dst = fresh dtype shape and idx = fresh Nx_dtype.Int64 shape in
      sort_with ~arg ~descending threads ~axis x ~dst ~idx;
      if arg then bytes_of idx else bytes_of dst
    in
    same_bytes (Printf.sprintf "%s %s" (Nx_dtype.to_string dtype) l.name) run
  in
  group "sorts"
    (List.concat_map
       (fun (arg, descending) ->
         List.map
           (fun (l, axis) ->
             case
               (Printf.sprintf "%s%s of %s along %d"
                  (if arg then "argsort" else "sort")
                  (if descending then " descending" else "")
                  l.name axis)
               (fun v ->
                 List.iter
                   (law ~arg ~descending l axis v)
                   [ D Float32; D Float64; D Int64 ]))
           layouts)
       [ (false, false); (false, true); (true, false); (true, true) ])

(* Groups: rows past one block of 2^16, so that the blocks' groups merge. Keys
   are drawn among [distinct] words, a few repeated in every block or most of
   them unique. *)

let groups =
  let layouts =
    [
      contiguous [| 200_000; 1 |];
      contiguous [| 140_000; 2 |];
      flipped (contiguous [| 140_000; 2 |]);
      transposed [| 3; 70_000 |];
    ]
  in
  let law l distinct (seed, _) =
    let st = Random.State.make [| seed |] in
    let n = Array.fold_left ( * ) 1 l.base in
    let buffer = Nx_array.Elements.create Nx_dtype.UInt64 n in
    let set = Nx_array.Elements.set Nx_dtype.UInt64 buffer in
    for i = 0 to n - 1 do
      set i (Random.State.int64 st distinct)
    done;
    let x =
      {
        Nx_array.dtype = Nx_dtype.UInt64;
        view = l.view (View.create l.base);
        buffer;
      }
    in
    let run threads =
      let dst = fresh Nx_dtype.Int64 [| (View.shape x.view).(0) |] in
      group_with threads x ~dst;
      bytes_of dst
    in
    same_bytes (Printf.sprintf "%Ld distinct words, %s" distinct l.name) run
  in
  group "groups"
    (List.map
       (fun l ->
         case ("group of " ^ l.name) (fun v ->
             List.iter (fun d -> law l d v) [ 3L; 5_000L; Int64.max_int ]))
       layouts)

let () =
  exit
    (run "nx.cpu planner" [ reductions; arg_reductions; scans; sorts; groups ])
