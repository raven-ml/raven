(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AMD GPUs as nx devices through the kernel driver, on real hardware: what
   Nx_amd opens is the runtime's GPU computed by nx.amd's backend, and only a
   call of [get_pci] or [device_pci] reaches PCI. The backend copies and casts
   every served dtype bit for bit as the host does, under every layout and for
   every value of the dtypes of 8 and 16 bits, from two domains at once, and
   refuses the rest naming the remedies. Tests needing a GPU skip without
   one. *)

open Windtrap
open Nx_test

let device = Testable.make ~pp:Nx.Device.pp ~equal:Nx.Device.equal
let memory = Testable.make ~pp:Nx_device.pp ~equal:Nx_device.equal
let gpu () = match Nx_amd.get 0 with Ok d -> d | Error e -> skip ~reason:e ()

let opening =
  group "opening"
    [
      test "device i is the runtime's GPU i of the kernel driver" (fun () ->
          let d = gpu () in
          let m = Result.get_ok (Nx_amd_device.get ~interface:Kernel 0) in
          equal device d (Nx_amd.device 0);
          equal memory m (Nx.Device.memory d);
          equal string (Nx_device.name m) (Nx.Device.name d);
          not_equal device Nx.Device.host d);
      test "get fails without touching PCI where the kernel driver is absent"
        (fun () ->
          if Sys.file_exists "/dev/kfd" then skip ~reason:"/dev/kfd exists" ();
          let why = require_error (Nx_amd.get 0) in
          raises (Failure why) (fun () -> ignore (Nx_amd.device 0)));
      test "the changes to the machine are the runtime's" (fun () ->
          let n = Nx_amd_device.count () in
          List.iter
            (fun (call, f, g) ->
              equal ~msg:call (result unit string) (g n) (f n))
            [
              ("detach", Nx_amd.detach, Nx_amd_device.detach);
              ("attach", Nx_amd.attach, Nx_amd_device.attach);
              ("reset", Nx_amd.reset, Nx_amd_device.reset);
              ( "fetch_firmware",
                Nx_amd.fetch_firmware,
                Nx_amd_device.fetch_firmware );
            ]);
      test "a negative index raises Invalid_argument" (fun () ->
          List.iter
            (fun f -> raises_match Exn.invalid_arg (fun () -> ignore (f (-1))))
            [ Nx_amd.get; Nx_amd.get_pci ];
          List.iter
            (fun f -> raises_match Exn.invalid_arg (fun () -> ignore (f (-1))))
            [ Nx_amd.device; Nx_amd.device_pci ]);
    ]

(* Computing *)

let on_gpu x = Nx.place (Nx.Placement.on (gpu ())) x
let host x = Nx.place Nx.Placement.host x
let placed f (Nx.P x) = (fun (Nx.P y) -> Nx.P (host y)) (f (Nx.P (on_gpu x)))

(* [f x] on the GPU, read back, and on the host, bit for bit. *)
let same f x =
  equal Stored.packed (f (Nx.P x))
    ((fun (Nx.P y) -> Nx.P (host y)) (f (Nx.P (on_gpu x))))

type dtype = Dtype : ('a, 'b) Nx.dtype -> dtype

let pp_dtype ppf (Dtype d) = Nx_dtype.pp ppf d

(* The dtypes the backend serves. *)
let served =
  [
    Dtype Nx.bool;
    Dtype Nx.int8;
    Dtype Nx.uint8;
    Dtype Nx.int16;
    Dtype Nx.uint16;
    Dtype Nx.int32;
    Dtype Nx.uint32;
    Dtype Nx.int64;
    Dtype Nx.uint64;
    Dtype Nx.float16;
    Dtype Nx.bfloat16;
    Dtype Nx.float32;
    Dtype Nx.float64;
    Dtype Nx.float8_e4m3;
    Dtype Nx.float8_e5m2;
  ]

let cast (Dtype d) (Nx.P x) = Nx.P (Nx.cast d x)
let copy (Nx.P x) = Nx.P (Nx.copy x)

let served_case (Stored.Case c) =
  List.exists (fun (Dtype d) -> Nx_dtype.to_string d = c.name) served

let conformance =
  group "conformance"
    (List.concat_map
       (fun (Stored.Case c) ->
         [
           prop (c.name ^ " values of every layout copy as on the host")
             c.tensors (fun x ->
               cover "strided" (not (Nx.is_c_contiguous x));
               cover "contiguous" (Nx.is_c_contiguous x && Nx.numel x > 1);
               same copy x);
           prop
             (c.name ^ " values of every layout cast as on the host")
             (Gen.pair c.tensors (Gen.of_list ~pp:pp_dtype served))
             (fun (x, d) ->
               cover "strided" (not (Nx.is_c_contiguous x));
               same (cast d) x);
         ])
       (List.filter served_case Stored.every))

(* Elementwise operations *)

(* Floats compare bit for bit but for NaN, whose sign and payload nx leaves
   unspecified off the host where an operation makes one. *)
let floats_or_bits =
  Testable.make ~pp:Stored.pp_packed
    ~equal:(fun (Nx.P a as pa) (Nx.P b as pb) ->
      if not (Nx_dtype.is_float (Nx.dtype a)) then
        Testable.equal Stored.packed pa pb
      else
        Nx.shape a = Nx.shape b
        && List.for_all2
             (fun x y ->
               (Float.is_nan x && Float.is_nan y)
               || Int64.bits_of_float x = Int64.bits_of_float y)
             (Array.to_list (Nx.to_array (Nx.cast Nx.float64 a)))
             (Array.to_list (Nx.to_array (Nx.cast Nx.float64 b))))

type unary = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
type binary = { g : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

type compare = {
  c : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t;
}

(* An operation of operands of one dtype, which broadcast: its name, the dtypes
   it takes by name, and whether its results are the host's bit for bit, NaN
   included, since it moves, compares or orders its operands, or computes on
   integers. *)
type op = {
  name : string;
  arity : int;
  takes : string -> bool;
  bits : bool;
  run : Nx.packed list -> Nx.packed;
}

let floats =
  [ "float16"; "bfloat16"; "float32"; "float64"; "float8_e4m3"; "float8_e5m2" ]

let is_float d = List.mem d floats
let numeric d = d <> "bool"
let integer d = not (is_float d || d = "bool")
let every _ = true

let unary name takes ~bits { f } =
  let run = function
    | [ Nx.P x ] -> Nx.P (f x)
    | _ -> invalid_arg "one operand"
  in
  { name; arity = 1; takes; bits; run }

let binary name takes ~bits { g } =
  let run = function
    | [ Nx.P a; b ] -> Nx.P (g a (Nx.unpack (Nx.dtype a) b))
    | _ -> invalid_arg "two operands"
  in
  { name; arity = 2; takes; bits; run }

let comparison name { c } =
  let run = function
    | [ Nx.P a; b ] -> Nx.P (c a (Nx.unpack (Nx.dtype a) b))
    | _ -> invalid_arg "two operands"
  in
  { name; arity = 2; takes = every; bits = true; run }

let ops =
  [
    unary "neg" numeric ~bits:true { f = Nx.neg };
    unary "recip" numeric ~bits:false { f = Nx.recip };
    unary "abs" numeric ~bits:true { f = Nx.abs };
    unary "sign" numeric ~bits:true { f = Nx.sign };
    unary "sqrt" is_float ~bits:false { f = Nx.sqrt };
    unary "trunc" numeric ~bits:false { f = Nx.trunc };
    unary "ceil" numeric ~bits:false { f = Nx.ceil };
    unary "floor" numeric ~bits:false { f = Nx.floor };
    unary "round" numeric ~bits:false { f = Nx.round };
    binary "add" numeric ~bits:false { g = Nx.add };
    binary "sub" numeric ~bits:false { g = Nx.sub };
    binary "mul" numeric ~bits:false { g = Nx.mul };
    binary "div" numeric ~bits:false { g = Nx.div };
    binary "mod" numeric ~bits:false { g = Nx.mod_ };
    binary "pow" integer ~bits:true { g = Nx.pow };
    binary "maximum" every ~bits:true { g = Nx.maximum };
    binary "minimum" every ~bits:true { g = Nx.minimum };
    binary "bitwise_and" (Fun.negate is_float) ~bits:true { g = Nx.bitwise_and };
    binary "bitwise_or" (Fun.negate is_float) ~bits:true { g = Nx.bitwise_or };
    binary "bitwise_xor" (Fun.negate is_float) ~bits:true { g = Nx.bitwise_xor };
    comparison "equal" { c = Nx.equal };
    comparison "not_equal" { c = Nx.not_equal };
    comparison "less" { c = Nx.less };
    comparison "less_equal" { c = Nx.less_equal };
    {
      name = "fma";
      arity = 3;
      takes = numeric;
      bits = false;
      run =
        (function
        | [ Nx.P a; b; c ] ->
            let u = Nx.unpack (Nx.dtype a) in
            Nx.P (Nx.fma a (u b) (u c))
        | _ -> invalid_arg "three operands");
    };
    {
      name = "where";
      arity = 3;
      takes = every;
      bits = true;
      run =
        (function
        | [ Nx.P a; b; c ] ->
            let u = Nx.unpack (Nx.dtype a) in
            Nx.P (Nx.where (Nx.less a (u b)) (u b) (u c))
        | _ -> invalid_arg "three operands");
    };
  ]

(* [op] of [xs] on the GPU, read back, against the host's. *)
let as_on_host op xs =
  let on (Nx.P x) = Nx.P (on_gpu x) and back (Nx.P y) = Nx.P (host y) in
  equal ~msg:op.name
    (if op.bits then Stored.packed else floats_or_bits)
    (op.run xs)
    (back (op.run (List.map on xs)))

(* Operands drawn as a value, its reverse and the value again: one shape, in two
   layouts. *)
let drawn_operands op (Nx.P x) =
  let r = Nx.P (Nx.flip x) in
  match op.arity with
  | 1 -> [ Nx.P x ]
  | 2 -> [ Nx.P x; r ]
  | _ -> [ Nx.P x; r; Nx.P x ]

let elementwise =
  group "elementwise"
    (List.map
       (fun (Stored.Case c) ->
         let mine = List.filter (fun op -> op.takes c.name) ops in
         prop ~count:400
           (c.name ^ " values of every layout, each operation's host result")
           (Gen.pair c.tensors
              (Gen.of_list
                 ~pp:(fun ppf op -> Format.pp_print_string ppf op.name)
                 mine))
           (fun (x, op) ->
             List.iter (fun o -> cover o.name (o == op)) mine;
             cover "strided" (not (Nx.is_c_contiguous x));
             as_on_host op (drawn_operands op (Nx.P x))))
       (List.filter served_case Stored.every))

(* Every value of each dtype of 8 or 16 bits, from the integers of its width. *)
let every_value (Dtype d) =
  match Nx_dtype.itemsize d with
  | 1 ->
      Nx.P (Nx.bitcast d (Nx.create Nx.uint8 [| 256 |] (Array.init 256 Fun.id)))
  | _ ->
      Nx.P
        (Nx.bitcast d
           (Nx.create Nx.uint16 [| 65536 |] (Array.init 65536 Fun.id)))

(* The dtypes of 8 and 16 bits whose every bit pattern is a value. *)
let narrow =
  List.filter
    (fun (Dtype d) ->
      Nx_dtype.itemsize d <= 2 && Nx_dtype.to_string d <> "bool")
    served

let sweep =
  let name (Dtype d) = Nx_dtype.to_string d in
  group "every value"
    (List.map
       (fun src ->
         test
           (Printf.sprintf "every %s cast to each dtype as on the host"
              (name src))
           (fun () ->
             let x = every_value src in
             List.iter
               (fun d ->
                 let msg = Format.asprintf "to %a" pp_dtype d in
                 equal ~msg Stored.packed (cast d x) (placed (cast d) x))
               served))
       narrow
    @ List.map
        (fun src ->
          test
            (Printf.sprintf
               "every %s through each unary operation as on the host" (name src))
            (fun () ->
              List.iter
                (fun op ->
                  if op.arity = 1 && op.takes (name src) then
                    as_on_host op [ every_value src ])
                ops))
        narrow
    @ List.filter_map
        (fun (Dtype d as src) ->
          if Nx_dtype.itemsize d <> 1 then None
          else
            Some
              (test
                 (Printf.sprintf
                    "every pair of %s through each binary operation as on the \
                     host"
                    (name src))
                 (fun () ->
                   let (Nx.P x) = every_value src in
                   let rows = Nx.P (Nx.reshape [| 256; 1 |] x)
                   and columns = Nx.P (Nx.reshape [| 1; 256 |] x) in
                   List.iter
                     (fun op ->
                       if op.arity = 2 && op.takes (name src) then
                         as_on_host op [ rows; columns ])
                     ops)))
        narrow)

(* A value of each served dtype: 64 elements of bytes drawn once, read as it;
   for bool, the bytes' low bits, which bool holds as they are. *)
let sample (Dtype d) =
  let k = Nx_dtype.itemsize d in
  let bytes =
    Nx.create Nx.uint8 [| 64; k |]
      (Array.init (64 * k) (fun i -> i * 37 mod 256))
  in
  match d with
  | Bool ->
      Nx.P
        (Nx.cast Nx.bool
           (Nx.reshape [| 64 |] (Nx.bitwise_and bytes (Nx.ones_like bytes))))
  | _ when k = 1 -> Nx.P (Nx.bitcast d (Nx.reshape [| 64 |] bytes))
  | _ -> Nx.P (Nx.bitcast d bytes)

(* The host's casts of the samples, computed once: a program on two domains
   replays the reference along each order of its calls. *)
let host_casts =
  lazy
    (List.map
       (fun s -> (s, List.map (fun d -> (d, cast d (sample s))) served))
       served)

let domains =
  let dtypes = Gen.of_list ~pp:pp_dtype served in
  let commands =
    [
      command "cast"
        (dtypes @-> dtypes @-> returns Stored.packed)
        (fun s d -> List.assq d (List.assq s (Lazy.force host_casts)))
        (fun s d -> placed (cast d) (sample s));
    ]
  in
  (* First in the run, before any other test loads a key: each program runs 50
     times, and its casts load fresh keys from both domains at once. *)
  group "domains"
    [
      stateful ~tags:[ "slow" ] ~count:10 ~domains:2
        "casts from two domains at once" commands;
      stateful "casts from one domain" commands;
    ]

let computing =
  group "computing"
    [
      test "values placed on it read back" (fun () ->
          let p = Nx.Placement.on (gpu ()) in
          let x = Nx.create Nx.float32 [| 3 |] [| 1.; -0.; 3.5 |] in
          let y = Nx.place p x in
          equal bool true (Nx.Placement.equal p (Nx.placement y));
          equal (array float_exact) (Nx.to_array x) (Nx.to_array y));
      test "the results of a cast are on the GPU" (fun () ->
          let p = Nx.Placement.on (gpu ()) in
          let y = Nx.cast Nx.int32 (Nx.place p (Nx.ones Nx.float32 [| 4 |])) in
          equal bool true (Nx.Placement.equal p (Nx.placement y)));
      test "an operation it has no kernel for raises, naming the remedies"
        (fun () ->
          let x = on_gpu (Nx.ones Nx.float32 [| 2 |]) in
          raises_match
            (Exn.invalid_arg ~substring:"no scans. Compile it with Rune.jit")
            (fun () -> ignore (Nx.cumsum x)));
      test "a dtype it does not serve raises, naming it" (fun () ->
          let x = on_gpu (Nx.ones Nx.complex64 [| 2 |]) in
          raises_match (Exn.invalid_arg ~substring:"no complex dtypes")
            (fun () -> ignore (Nx.cast Nx.float32 x)));
      test "a value over a bare GPU buffer computes once placed on the device"
        (fun () ->
          let d = gpu () in
          let b =
            Nx_device.Buffer.create (Nx.Device.memory d) Nx_dtype.Scalar.Float32
              4
          in
          let x = Nx.of_buffer Nx.float32 [| 4 |] b in
          raises_match (Exn.invalid_arg ~substring:"has no eager kernels")
            (fun () -> ignore (Nx.cast Nx.int32 x));
          let y = Nx.place (Nx.Placement.on d) x in
          equal (array int) [| 4 |] (Nx.shape (Nx.cast Nx.int32 y)));
      test "a value of no element casts to one of no element" (fun () ->
          let y = Nx.cast Nx.int8 (on_gpu (Nx.zeros Nx.float32 [| 0; 3 |])) in
          equal (array int) [| 0; 3 |] (Nx.shape y));
    ]

(* Reductions *)

(* A reduction over the axes [axes] of a value: an extreme, whose result is the
   host's bit for bit, or a sum or a product. An argument reduction takes the
   first of [axes], or flattens the value when there is none. *)
type reduction = {
  title : string;
  extreme : bool;
  r : 'a 'b. int list -> ('a, 'b) Nx.t -> Nx.packed;
}

let arg f axes x =
  match axes with
  | [] -> Nx.P (f ?axis:None x)
  | a :: _ -> Nx.P (f ?axis:(Some a) x)

let reductions =
  [
    {
      title = "sum";
      extreme = false;
      r = (fun axes x -> Nx.P (Nx.sum ~axes x));
    };
    {
      title = "prod";
      extreme = false;
      r = (fun axes x -> Nx.P (Nx.prod ~axes x));
    };
    { title = "max"; extreme = true; r = (fun axes x -> Nx.P (Nx.max ~axes x)) };
    { title = "min"; extreme = true; r = (fun axes x -> Nx.P (Nx.min ~axes x)) };
    {
      title = "argmax";
      extreme = true;
      r = (fun axes x -> arg (fun ?axis x -> Nx.argmax ?axis x) axes x);
    };
    {
      title = "argmin";
      extreme = true;
      r = (fun axes x -> arg (fun ?axis x -> Nx.argmin ?axis x) axes x);
    };
  ]

let extremes = List.filter (fun red -> red.extreme) reductions
let sum = List.find (fun red -> red.title = "sum") reductions
let prod = List.find (fun red -> red.title = "prod") reductions

(* Whether [red] of [x] over [axes] has a value: an extreme needs elements. *)
let defined red axes x =
  (not red.extreme)
  ||
  match (red.title, axes) with
  | ("argmax" | "argmin"), [] -> Nx.numel x > 0
  | ("argmax" | "argmin"), a :: _ -> Nx.dim a x > 0
  | _ -> List.for_all (fun a -> Nx.dim a x > 0) axes

(* [red] of [x] on the GPU, read back, against the host's, bit for bit. Over
   several axes, nx states only that the host's max and min take one of the
   NaNs, so a NaN result there is any NaN. *)
let reduces_as_on_host ?msg red axes (Nx.P x) =
  let several =
    List.length axes > 1 && (red.title = "max" || red.title = "min")
  in
  equal ?msg
    (if several then floats_or_bits else Stored.packed)
    (red.r axes x)
    (placed (fun (Nx.P x) -> red.r axes x) (Nx.P x))

(* The axes of a value of rank [n] that the bits of [mask] select. *)
let masked mask n =
  List.filter (fun a -> mask land (1 lsl a) <> 0) (List.init n Fun.id)

(* Corner values under every layout, over drawn axes: the extremes of every
   dtype, NaNs and zeros of both signs among them, and the integer sums and
   products, which wrap whatever their association. *)
let reduced =
  group "reduced"
    (List.map
       (fun (Stored.Case c) ->
         let mine =
           List.filter (fun red -> red.extreme || integer c.name) reductions
         in
         prop ~count:400
           (c.name
          ^ " values of every layout, reduced over drawn axes as on the host")
           (Gen.triple c.tensors (Gen.int_range 0 255)
              (Gen.of_list
                 ~pp:(fun ppf red -> Format.pp_print_string ppf red.title)
                 mine))
           (fun (x, mask, red) ->
             let axes = masked mask (Nx.ndim x) in
             List.iter (fun r -> cover r.title (r == red)) mine;
             cover "strided" (not (Nx.is_c_contiguous x));
             cover "several axes" (List.length axes > 1);
             if defined red axes x then reduces_as_on_host red axes (Nx.P x)
             else
               raises_match Exn.invalid_arg (fun () ->
                   ignore (red.r axes (on_gpu x)))))
       (List.filter served_case Stored.every))

(* Shapes and the axes they reduce, whose rows cross the kernels' geometry: rows
   of a workgroup's threads and more, many rows walked by the grid, rows of
   strided elements, and a few long rows folded in two passes. *)
let geometries =
  [
    ([| 1000; 257 |], [ 1 ]);
    ([| 257; 1000 |], [ 0 ]);
    ([| 3; 100_000 |], [ 1 ]);
    ([| 100_000; 3 |], [ 0 ]);
    ([| 300; 301 |], [ 0; 1 ]);
    ([| 5; 7; 4099 |], [ 0; 2 ]);
  ]

(* Each geometry as it is laid out and transposed, which makes the reduced
   elements of a contiguous row strided and the reverse. *)
let laid_out =
  List.concat_map
    (fun (shape, axes) ->
      let n = Array.length shape in
      let t = Array.init n (fun i -> shape.(n - 1 - i)) in
      [
        ("contiguous", shape, Fun.id, axes);
        ( "transposed",
          t,
          (fun (Nx.P x) -> Nx.P (Nx.transpose x)),
          List.map (fun a -> n - 1 - a) axes );
      ])
    geometries

(* Values of [d] of shape [shape]: [f i] at the [i]th element in C order,
   computed at float64 and cast. *)
let values (Dtype d) shape f =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.P (Nx.cast d (Nx.create Nx.float64 shape (Array.init n f)))

let spread i = i * 7919 mod 13

(* Rows of many equal extremes; with NaNs of both signs; of zeros of both signs
   below the negative numbers' ties; of infinities of both signs. *)
let patterns =
  [
    ("ties", fun i -> Float.of_int (spread i - 6));
    ( "NaNs",
      fun i ->
        if i mod 997 = 3 then Float.nan
        else if i mod 1009 = 5 then Float.neg Float.nan
        else Float.of_int (spread i - 6) );
    ("zeros", fun i -> [| -0.; 0.; -1.; -0.; -2. |].(spread i mod 5));
    ( "infinities",
      fun i ->
        [| 1.; Float.neg_infinity; Float.infinity; -1.; 2. |].(spread i mod 5)
    );
  ]

(* Factors whose products are exact: ones of both signs, with a 2 and, in some
   rows, a zero of either sign. *)
let factors i =
  if i mod 4099 = 17 then 2.
  else if i mod 8191 = 4 then if i mod 2 = 0 then -0. else 0.
  else if spread i mod 5 = 0 then -1.
  else 1.

let geometry =
  group "geometry"
    (List.map
       (fun (Dtype d as dt) ->
         let name = Nx_dtype.to_string d in
         test
           (name
          ^ " rows of every length reduce as on the host, sums and products of \
             exact partials included") (fun () ->
             List.iter
               (fun (layout, shape, lay, axes) ->
                 let msg red what =
                   let ints a = String.concat "," (List.map string_of_int a) in
                   Printf.sprintf "%s of %s [%s] over %s (%s)" red.title what
                     (ints (Array.to_list shape))
                     (ints axes) layout
                 in
                 List.iter
                   (fun (what, f) ->
                     let x = lay (values dt shape f) in
                     List.iter
                       (fun red ->
                         reduces_as_on_host ~msg:(msg red what) red axes x)
                       extremes)
                   patterns;
                 if name <> "bool" then begin
                   reduces_as_on_host ~msg:(msg sum "ties") sum axes
                     (lay (values dt shape (List.assoc "ties" patterns)));
                   reduces_as_on_host ~msg:(msg prod "factors") prod axes
                     (lay (values dt shape factors))
                 end)
               laid_out))
       served)

let float_dtypes = List.filter (fun (Dtype d) -> Nx_dtype.is_float d) served

let sums =
  group "sums"
    [
      test "sums of nothing are 0 and products of nothing 1, at every dtype"
        (fun () ->
          List.iter
            (fun (Dtype d) ->
              if Nx_dtype.to_string d <> "bool" then
                List.iter
                  (fun (shape, axes, out) ->
                    let x = on_gpu (Nx.zeros d shape) in
                    let msg = Nx_dtype.to_string d in
                    equal ~msg Stored.packed
                      (Nx.P (Nx.zeros d out))
                      (Nx.P (host (Nx.sum ~axes x)));
                    equal ~msg Stored.packed
                      (Nx.P (Nx.ones d out))
                      (Nx.P (host (Nx.prod ~axes x))))
                  [
                    ([| 3; 0 |], [ 1 ], [| 3 |]);
                    ([| 0; 5 |], [ 0 ], [| 5 |]);
                    ([| 0 |], [ 0 ], [||]);
                  ])
            served);
      test "sums of -0 are +0, at every float dtype and length" (fun () ->
          List.iter
            (fun (Dtype d) ->
              List.iter
                (fun n ->
                  let x =
                    on_gpu (Nx.cast d (Nx.full Nx.float64 [| 2; n |] (-0.)))
                  in
                  equal
                    ~msg:
                      (Printf.sprintf "%s, %d terms" (Nx_dtype.to_string d) n)
                    Stored.packed
                    (Nx.P (Nx.zeros d [| 2 |]))
                    (Nx.P (host (Nx.sum ~axes:[ 1 ] x))))
                [ 1; 5; 300; 5000; 100_000 ])
            float_dtypes);
      cases
        ~name:(fun (title, _, _) -> title)
        "a sum of a NaN is NaN, of one infinity that infinity, of both NaN"
        [
          ("a NaN among numbers", [| 1.; Float.nan; 2. |], Float.nan);
          ("+inf among numbers", [| 1.; Float.infinity; 2. |], Float.infinity);
          ( "-inf among numbers",
            [| -1.; Float.neg_infinity; 2. |],
            Float.neg_infinity );
          ( "both infinities",
            [| Float.infinity; 1.; Float.neg_infinity |],
            Float.nan );
        ]
        (fun (_, xs, expected) ->
          List.iter
            (fun (Dtype d) ->
              if Nx_dtype.to_string d <> "float8_e4m3" || Float.is_nan expected
              then begin
                let x = Nx.cast d (Nx.create Nx.float64 [| 3 |] xs) in
                let got =
                  Nx.item [] (Nx.cast Nx.float64 (host (Nx.sum (on_gpu x))))
                in
                let msg = Nx_dtype.to_string d in
                if Float.is_nan expected then
                  equal ~msg bool true (Float.is_nan got)
                else equal ~msg float_exact expected got
              end)
            float_dtypes);
    ]

(* Error bounds *)

(* A float dtype's format: the bits of its significand, its least positive and
   greatest finite values, and the unit roundoff of the format it sums in. *)
type format = { m : int; tiny : float; top : float; eps : float }

let format : type a b. (a, b) Nx.dtype -> format = function
  | Float16 -> { m = 10; tiny = 0x1p-24; top = 65504.; eps = 0x1p-24 }
  | BFloat16 -> { m = 7; tiny = 0x1p-133; top = 0x1.fep127; eps = 0x1p-24 }
  | Float32 -> { m = 23; tiny = 0x1p-149; top = 0x1.fffffep127; eps = 0x1p-24 }
  | Float64 ->
      { m = 52; tiny = 0x1p-1074; top = Float.max_float; eps = 0x1p-53 }
  | Float8_e4m3 -> { m = 3; tiny = 0x1p-9; top = 448.; eps = 0x1p-24 }
  | Float8_e5m2 -> { m = 2; tiny = 0x1p-16; top = 57344.; eps = 0x1p-24 }
  | _ -> invalid_arg "not a float dtype"

(* The sum of [xs] rounded once: Neumaier's compensated sum, whose error is a
   rounding of the result plus terms of the order of the squared roundoff. *)
let exact_sum xs =
  let s = ref 0. and c = ref 0. in
  Array.iter
    (fun x ->
      let t = !s +. x in
      c :=
        !c +. if Float.abs !s >= Float.abs x then !s -. t +. x else x -. t +. !s;
      s := t)
    xs;
  !s +. !c

(* The spacing of the floats of format [f] at the magnitude [v]. *)
let ulp f v =
  if v = 0. then f.tiny
  else Float.max f.tiny (Float.ldexp 1. (snd (Float.frexp v) - 1 - f.m))

let bounded =
  group "error bounds"
    [
      prop ~count:200
        "a float sum of n terms lies within n eps sum |x| of the exact sum, \
         plus a rounding to a narrow dtype"
        Gen.(
          quad
            (of_list ~pp:pp_dtype float_dtypes)
            (int_range 1 6) (int_range 0 20_000) (pair bool int))
        (fun (Dtype d, rows, n, (transposed, seed)) ->
          let f = format d in
          let rng = Random.State.make [| seed |] in
          let scale = f.top /. Float.of_int (4 * Int.max n 1) in
          let x =
            Nx.cast d
              (Nx.create Nx.float64 [| rows; n |]
                 (Array.init (rows * n) (fun _ ->
                      (Random.State.float rng 2. -. 1.)
                      *. Float.ldexp scale (-Random.State.int rng 8))))
          in
          let terms = Nx.to_array (Nx.cast Nx.float64 x) in
          let got =
            if transposed then
              Nx.sum ~axes:[ 0 ] (on_gpu (Nx.contiguous (Nx.transpose x)))
            else Nx.sum ~axes:[ 1 ] (on_gpu x)
          in
          let got = Nx.to_array (Nx.cast Nx.float64 (host got)) in
          cover "two passes" (rows = 1 && n > 4096);
          cover "transposed" transposed;
          for r = 0 to rows - 1 do
            let row = Array.sub terms (r * n) n in
            let s = exact_sum row in
            let e =
              Float.of_int n *. f.eps
              *. Array.fold_left (fun a x -> a +. Float.abs x) 0. row
            in
            let narrow = if f.m < 23 then ulp f (Float.abs s +. e) else 0. in
            at_most
              ~msg:(Printf.sprintf "row %d of %d terms, exact %h" r n s)
              float_exact
              ~than:(e +. narrow +. (Float.epsilon *. Float.abs s))
              (Float.abs (got.(r) -. s))
          done);
    ]

(* Matrix products *)

(* How an operand holds its matrices: as stored, transposed, as every other
   column of a wider value, or as one row broadcast down the rows. *)
type layout = Stored | Transposed | Strided | Broadcast

let layouts = [ Stored; Transposed; Strided; Broadcast ]

let pp_layout ppf l =
  Format.pp_print_string ppf
    (match l with
    | Stored -> "stored"
    | Transposed -> "transposed"
    | Strided -> "strided"
    | Broadcast -> "broadcast")

(* Values of [d] of shape [shape] from [seed]: integers drawn over the whole
   range, which wrap in products; floats small integers, whose products and sums
   are exact, with an infinity or a NaN here and there. *)
let drawn (Dtype d as dt) shape seed =
  let rng = Random.State.make [| seed |] in
  let n = Array.fold_left ( * ) 1 shape in
  if integer (Nx_dtype.to_string d) then
    Nx.P
      (Nx.cast d
         (Nx.create Nx.int64 shape
            (Array.init n (fun _ -> Random.State.bits64 rng))))
  else
    values dt shape (fun _ ->
        match Random.State.int rng 200 with
        | 0 -> Float.infinity
        | 1 -> Float.neg_infinity
        | 2 -> Float.nan
        | v -> Float.of_int ((v mod 7) - 3))

(* An operand of shape [shape] laid out as [l], its values from [fill]. *)
let operand l shape fill =
  let n = Array.length shape in
  let rows = shape.(n - 2) and cols = shape.(n - 1) in
  let with_ i v = Array.mapi (fun j x -> if j = i then v else x) shape in
  match l with
  | Stored -> fill shape
  | Transposed ->
      let (Nx.P x) = fill shape in
      Nx.P
        (Nx.swapaxes (n - 1) (n - 2)
           (Nx.contiguous (Nx.swapaxes (n - 1) (n - 2) x)))
  | Strided when cols > 0 ->
      let (Nx.P w) = fill (with_ (n - 1) (2 * cols)) in
      Nx.P
        (Nx.squeeze ~axes:[ -1 ]
           (Nx.sliding_window ~axis:(-1) ~window:1 ~step:2 w))
  | Broadcast when rows > 0 ->
      let (Nx.P x) = fill (with_ (n - 2) 1) in
      Nx.P (Nx.broadcast_to shape x)
  | Strided | Broadcast -> fill shape

(* The batch shapes of two operands: equal, one operand's absent, one holding a
   single matrix along an axis, and an empty batch. *)
let batches =
  [
    ([||], [||]);
    ([| 3 |], [||]);
    ([||], [| 2 |]);
    ([| 2; 1 |], [| 3 |]);
    ([| 1 |], [| 4 |]);
    ([| 2; 3 |], [| 2; 3 |]);
    ([| 0 |], [| 1 |]);
  ]

(* Extents on both sides of the kernels' tiles, of 64 rows and columns and 16
   along k. *)
let extents = [ 0; 1; 2; 3; 16; 17; 64; 65; 130 ]

type product = {
  batch : int array * int array;
  m : int;
  k : int;
  n : int;
  lay : layout * layout;
  seed : int;
}

let pp_product ppf p =
  let dims a = String.concat "x" (List.map string_of_int (Array.to_list a)) in
  Format.fprintf ppf "[%s] %dx%d (%a) times [%s] %dx%d (%a), seed %d"
    (dims (fst p.batch))
    p.m p.k pp_layout (fst p.lay)
    (dims (snd p.batch))
    p.k p.n pp_layout (snd p.lay) p.seed

let products =
  Gen.with_pp pp_product
    Gen.(
      let dim = of_list ~pp:Format.pp_print_int extents
      and layout = of_list ~pp:pp_layout layouts in
      let+ batch = of_list ~pp:(fun _ _ -> ()) batches
      and+ m = dim
      and+ k = dim
      and+ n = dim
      and+ lay = pair layout layout
      and+ seed = int in
      { batch; m; k; n; lay; seed })

(* The operands of [p] at [d]. *)
let operands dt p =
  let a =
    operand (fst p.lay)
      (Array.append (fst p.batch) [| p.m; p.k |])
      (fun s -> drawn dt s p.seed)
  and b =
    operand (snd p.lay)
      (Array.append (snd p.batch) [| p.k; p.n |])
      (fun s -> drawn dt s (p.seed + 1))
  in
  (a, b)

let matmul (Nx.P a) b = Nx.P (Nx.matmul a (Nx.unpack (Nx.dtype a) b))
let gpu (Nx.P x) = Nx.P (on_gpu x)

let multiplied =
  group "multiplied"
    (List.filter_map
       (fun (Dtype d as dt) ->
         let name = Nx_dtype.to_string d in
         if name = "bool" then None
         else
           Some
             (prop ~count:150
                (name
               ^ " products of every layout and batch broadcast, as on the \
                  host where their sums are exact") products (fun p ->
                  let a, b = operands dt p in
                  cover "a transposed operand"
                    (fst p.lay = Transposed || snd p.lay = Transposed);
                  cover "a strided operand"
                    (fst p.lay = Strided || snd p.lay = Strided);
                  cover "a broadcast batch" (fst p.batch <> snd p.batch);
                  cover "an empty contraction" (p.k = 0);
                  cover "several tiles" ((p.m > 64 || p.n > 64) && p.k > 16);
                  equal
                    (if Nx_dtype.is_float d then floats_or_bits
                     else Stored.packed)
                    (matmul a b)
                    ((fun (Nx.P y) -> Nx.P (host y)) (matmul (gpu a) (gpu b))))))
       served)

(* The product of [xa] of shape [sa] and [xb] of shape [sb] at [dt], computed on
   the GPU and read back. *)
let gpu_product dt (sa, xa) (sb, xb) =
  let on (sh, xs) =
    let (Nx.P x) = values dt sh (fun i -> xs.(i)) in
    Nx.P (on_gpu x)
  in
  let (Nx.P y) = matmul (on (sa, xa)) (on (sb, xb)) in
  Nx.P (host y)

let product_cases =
  group "products"
    [
      test "an empty contraction is 0, at every numeric dtype" (fun () ->
          List.iter
            (fun (Dtype d as dt) ->
              let msg = Nx_dtype.to_string d in
              if msg <> "bool" then
                equal ~msg Stored.packed
                  (Nx.P (Nx.zeros d [| 3; 2 |]))
                  (gpu_product dt ([| 3; 0 |], [||]) ([| 0; 2 |], [||])))
            served);
      test "products summing to exactly zero, or of -0, give +0" (fun () ->
          List.iter
            (fun (Dtype d as dt) ->
              List.iter
                (fun xs ->
                  equal ~msg:(Nx_dtype.to_string d) Stored.packed
                    (Nx.P (Nx.zeros d [| 1; 1 |]))
                    (gpu_product dt ([| 1; 2 |], xs) ([| 2; 1 |], [| 2.; 2. |])))
                [ [| 1.; -1. |]; [| -0.; -0. |] ])
            float_dtypes);
      test "a product without rows, columns or matrices has none" (fun () ->
          List.iter
            (fun (sa, sb, out) ->
              let a = on_gpu (Nx.ones Nx.float32 sa)
              and b = on_gpu (Nx.ones Nx.float32 sb) in
              equal (array int) out (Nx.shape (host (Nx.matmul a b))))
            [
              ([| 0; 3 |], [| 3; 2 |], [| 0; 2 |]);
              ([| 2; 3 |], [| 3; 0 |], [| 2; 0 |]);
              ([| 0; 2; 3 |], [| 3; 2 |], [| 0; 2; 2 |]);
            ]);
      cases
        ~name:(fun (Dtype d, _, _) -> Nx_dtype.to_string d)
        "a narrow float's products sum at float32 and round once"
        [
          (Dtype Nx.float16, 2048., 2052.);
          (Dtype Nx.bfloat16, 256., 260.);
          (Dtype Nx.float8_e4m3, 16., 20.);
          (Dtype Nx.float8_e5m2, 16., 20.);
        ]
        (fun ((Dtype d as dt), big, expected) ->
          (* big + 3 rounds once to [expected]; a sum at the dtype stays at
             [big], each 1 falling below half its spacing there. *)
          equal Stored.packed
            (Nx.P (Nx.cast d (Nx.create Nx.float64 [| 1; 1 |] [| expected |])))
            (gpu_product dt
               ([| 1; 4 |], [| big; 1.; 1.; 1. |])
               ([| 4; 1 |], [| 1.; 1.; 1.; 1. |])));
    ]

(* The exact sum of the products of [xs] and [ys], rounded once, and the sum of
   their magnitudes: each product as a sum of two floats by a fused
   multiply-add, then the compensated sum of the pieces. *)
let exact_dot xs ys =
  let n = Array.length xs in
  let pieces =
    Array.init (2 * n) (fun i ->
        let x = xs.(i / 2) and y = ys.(i / 2) in
        let p = x *. y in
        if i mod 2 = 0 then p else Float.fma x y (-.p))
  in
  ( exact_sum pieces,
    Array.fold_left ( +. ) 0.
      (Array.init n (fun i -> Float.abs (xs.(i) *. ys.(i)))) )

let product_bounds =
  group "product bounds"
    [
      prop ~count:150
        "a float product's element lies within k eps sum |a b| of the exact \
         one, plus a rounding to a narrow dtype"
        Gen.(
          quad
            (of_list ~pp:pp_dtype float_dtypes)
            (triple (int_range 1 40) (int_range 0 700) (int_range 1 40))
            (pair
               (of_list ~pp:pp_layout layouts)
               (of_list ~pp:pp_layout layouts))
            int)
        (fun ((Dtype d as dt), (m, k, n), (la, lb), seed) ->
          let f = format d in
          let rng = Random.State.make [| seed |] in
          let scale = Float.sqrt (f.top /. Float.of_int (4 * Int.max k 1)) in
          let fill s =
            values dt s (fun _ ->
                (Random.State.float rng 2. -. 1.)
                *. Float.ldexp scale (-Random.State.int rng 6))
          in
          let (Nx.P a) = operand la [| m; k |] fill in
          let b = Nx.unpack (Nx.dtype a) (operand lb [| k; n |] fill) in
          let wide x = Nx.to_array (Nx.cast Nx.float64 x) in
          let xa = wide a and xb = wide b in
          let got = wide (host (Nx.matmul (on_gpu a) (on_gpu b))) in
          cover "several k tiles" (k > 16);
          for i = 0 to m - 1 do
            for j = 0 to n - 1 do
              let s, mag =
                exact_dot
                  (Array.init k (fun l -> xa.((i * k) + l)))
                  (Array.init k (fun l -> xb.((l * n) + j)))
              in
              let e = Float.of_int k *. f.eps *. mag in
              let narrow = if f.m < 23 then ulp f (Float.abs s +. e) else 0. in
              at_most
                ~msg:
                  (Printf.sprintf "element (%d, %d) of %d terms, exact %h" i j k
                     s)
                float_exact
                ~than:(e +. narrow +. (Float.epsilon *. Float.abs s))
                (Float.abs (got.((i * n) + j) -. s))
            done
          done);
    ]

let accuracy = group "accuracy" (Nx_test.Accuracy.groups { put = on_gpu })

let () =
  exit
    (run "nx.amd"
       [
         domains;
         opening;
         computing;
         conformance;
         elementwise;
         sweep;
         reduced;
         geometry;
         sums;
         bounded;
         multiplied;
         product_cases;
         product_bounds;
         accuracy;
       ])
