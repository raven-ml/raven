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
  (* First in the run, before any other test loads a kernel: each program runs
     50 times, and its casts load fresh kernels from both domains at once. *)
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
          let x = on_gpu (Nx.ones Nx.float32 [| 1; 1 |]) in
          raises_match
            (Exn.invalid_arg ~substring:"no cholesky. Compile it with Rune.jit")
            (fun () -> ignore (Nx.cholesky x)));
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

(* Moves *)

(* [f x] with [x] on the GPU, read back, against [f x] on the host, bit for
   bit. *)
let moved ?msg f x =
  equal ?msg Stored.packed (Nx.P (f x)) (Nx.P (host (f (on_gpu x))))

(* Positions along an axis of [n]: inside it mostly, its bounds' neighbours, and
   the extremes of int64. *)
let positions n =
  Gen.frequency
    [
      (6, Gen.map Int64.of_int (Gen.int_range 0 (Int.max 0 (n - 1))));
      (2, Gen.map Int64.of_int (Gen.int_range (-2) (n + 2)));
      ( 1,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "%Ld")
          [ Int64.min_int; Int64.max_int ] );
    ]

(* A value drawn under every layout, of rank 1 at least, and one of its axes. *)
let with_axis tensors =
  Gen.bind tensors (fun x ->
      if Nx.ndim x = 0 then Gen.constant (Nx.reshape [| 1 |] x, 0)
      else Gen.map (fun a -> (x, a)) (Gen.int_range 0 (Nx.ndim x - 1)))

let moves =
  group "moves"
    (List.concat_map
       (fun (Stored.Case c) ->
         [
           prop
             (c.name
            ^ " values of every layout gather along an axis as on the host, \
               out-of-range indices included")
             (Gen.bind (with_axis c.tensors) (fun (x, axis) ->
                  let shape = Array.copy (Nx.shape x) in
                  let open Gen in
                  let* k = int_range 0 3 in
                  shape.(axis) <- k;
                  let+ ix =
                    array
                      ~size:(constant (Array.fold_left ( * ) 1 shape))
                      (positions (Nx.dim axis x))
                  in
                  (x, axis, Nx.create Nx.int64 shape ix)))
             (fun (x, axis, indices) ->
               let n = Int64.of_int (Nx.dim axis x) in
               cover "strided" (not (Nx.is_c_contiguous x));
               cover "out of range"
                 (Array.exists
                    (fun i -> i < 0L || i >= n)
                    (Nx.to_array indices));
               equal Stored.packed
                 (Nx.P (Nx.take_along_axis ~axis ~indices x))
                 (Nx.P
                    (host
                       (Nx.take_along_axis ~axis ~indices:(on_gpu indices)
                          (on_gpu x)))));
           prop
             (c.name ^ " values of every layout concatenate as on the host")
             (Gen.pair (with_axis c.tensors) (Gen.int_range 0 4))
             (fun ((x, axis), k) ->
               let k = Int.min k (Nx.dim axis x) in
               let head =
                 Array.mapi (fun d n -> if d = axis then (0, k) else (0, n))
               in
               cover "strided" (not (Nx.is_c_contiguous x));
               cover "an empty member" (k = 0 || Nx.numel x = 0);
               moved
                 (fun x ->
                   Nx.concatenate ~axis
                     [ x; Nx.shrink (head (Nx.shape x)) x; Nx.flip x ])
                 x);
           prop
             (c.name ^ " values of every layout pad with a value as on the host")
             (Gen.triple c.tensors c.tensors
                (Gen.array ~size:(Gen.constant 8)
                   (Gen.pair (Gen.int_range 0 3) (Gen.int_range 0 3))))
             (fun (x, y, widths) ->
               let widths = Array.sub widths 0 (Nx.ndim x) in
               cover "strided" (not (Nx.is_c_contiguous x));
               cover "empty" (Nx.numel x = 0);
               if Nx.numel y > 0 then
                 moved (Nx.pad widths (Nx.to_array y).(0)) x);
           prop
             (c.name ^ " values of every layout set at a window as on the host")
             (Gen.bind (Gen.pair c.tensors Gen.int) (fun (x, seed) ->
                  let rng = Random.State.make [| seed |] in
                  let window =
                    Array.map
                      (fun n ->
                        let a = Random.State.int rng (n + 1) in
                        (a, a + Random.State.int rng (n - a + 1)))
                      (Nx.shape x)
                  in
                  Gen.constant (x, window)))
             (fun (x, window) ->
               let specs =
                 Array.to_list (Array.map (fun (a, b) -> Nx.R (a, b)) window)
               in
               cover "strided" (not (Nx.is_c_contiguous x));
               cover "an empty window"
                 (Array.exists (fun (a, b) -> a = b) window);
               moved (fun x -> Nx.set specs (Nx.slice specs (Nx.flip x)) x) x);
         ])
       (List.filter served_case Stored.every))

let move_cases =
  group "move cases"
    [
      test "gathers out of range read zero, at every dtype" (fun () ->
          List.iter
            (fun (Dtype d) ->
              let x = on_gpu (Nx.ones d [| 3 |]) in
              let indices =
                on_gpu
                  (Nx.create Nx.int64 [| 6 |]
                     [| -1L; 3L; Int64.min_int; Int64.max_int; 0L; 2L |])
              in
              let ones = Nx.ones d [| 1 |] and zeros = Nx.zeros d [| 4 |] in
              equal ~msg:(Nx_dtype.to_string d) Stored.packed
                (Nx.P (Nx.concatenate ~axis:0 [ zeros; ones; ones ]))
                (Nx.P (host (Nx.take ~indices x))))
            served);
      test "a window at a position read on the GPU is clamped to fit" (fun () ->
          let x = Nx.create Nx.int32 [| 5 |] [| 0l; 1l; 2l; 3l; 4l |] in
          let v = Nx.create Nx.int32 [| 2 |] [| 7l; 8l |] in
          List.iter
            (fun (start, at) ->
              let s = on_gpu (Nx.scalar Nx.int64 start) in
              equal
                ~msg:(Printf.sprintf "start %Ld" start)
                Stored.packed
                (Nx.P (Nx.set [ R (at, at + 2) ] v x))
                (Nx.P (host (Nx.set [ D (s, 2) ] (on_gpu v) (on_gpu x)))))
            [ (0L, 0); (2L, 2); (3L, 3); (10L, 3); (-4L, 0) ]);
      test "negative pads raise, as on the host" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              ignore
                (Nx.pad [| (-1, 0) |] 0. (on_gpu (Nx.ones Nx.float32 [| 3 |])))));
    ]

(* Scans *)

(* A running value along an axis: an extreme, whose results are the host's bit
   for bit, or a sum or a product. *)
type running = {
  name : string;
  ordered : bool;
  f : 'a 'b. int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t;
}

let runnings =
  [
    { name = "cumsum"; ordered = false; f = (fun axis x -> Nx.cumsum ~axis x) };
    {
      name = "cumprod";
      ordered = false;
      f = (fun axis x -> Nx.cumprod ~axis x);
    };
    { name = "cummax"; ordered = true; f = (fun axis x -> Nx.cummax ~axis x) };
    { name = "cummin"; ordered = true; f = (fun axis x -> Nx.cummin ~axis x) };
  ]

let cumsum = List.find (fun r -> r.name = "cumsum") runnings
let cumprod = List.find (fun r -> r.name = "cumprod") runnings

(* [run] of [x] along [axis] on the GPU, read back, against the host's: bit for
   bit, but for a NaN of a sum or a product, which is any NaN. *)
let runs_as_on_host ?msg run axis x =
  let float = Nx_dtype.is_float (Nx.dtype x) in
  equal ?msg
    (if run.ordered || not float then Stored.packed else floats_or_bits)
    (Nx.P (run.f axis x))
    (Nx.P (host (run.f axis (on_gpu x))))

let scanned =
  group "scanned"
    (List.map
       (fun (Stored.Case c) ->
         let mine =
           List.filter (fun r -> r.ordered || c.name <> "bool") runnings
         in
         prop ~count:300
           (c.name ^ " values of every layout run along an axis as on the host")
           (Gen.pair (with_axis c.tensors)
              (Gen.of_list
                 ~pp:(fun ppf r -> Format.pp_print_string ppf r.name)
                 mine))
           (fun ((x, axis), run) ->
             List.iter (fun r -> cover r.name (r == run)) mine;
             cover "strided" (not (Nx.is_c_contiguous x));
             cover "an empty axis" (Nx.dim axis x = 0);
             cover "a one-element axis" (Nx.dim axis x = 1);
             runs_as_on_host run axis x))
       (List.filter served_case Stored.every))

(* Shapes and the axis they run along, crossing the kernels' geometry: short
   rows on a thread each, rows on a workgroup each, and few long rows in parts
   that start from the folds of the parts before them. *)
let scan_geometries =
  [
    ([| 200_000; 3 |], 1);
    ([| 1000; 257 |], 1);
    ([| 257; 1000 |], 0);
    ([| 5; 4099 |], 1);
    ([| 3; 100_000 |], 1);
    ([| 100_000; 3 |], 0);
    ([| 1; 300_000 |], 1);
  ]

let scan_geometry =
  group "scan geometry"
    (List.map
       (fun (Dtype d as dt) ->
         let name = Nx_dtype.to_string d in
         test
           (name
          ^ " rows of every length run as on the host, sums and products of \
             exact partials included") (fun () ->
             List.iter
               (fun (shape, axis) ->
                 let lay (Nx.P x) = function
                   | "transposed" -> Nx.P (Nx.transpose x)
                   | _ -> Nx.P x
                 in
                 List.iter
                   (fun layout ->
                     let t = layout = "transposed" in
                     let stored =
                       if t then [| shape.(1); shape.(0) |] else shape
                     and axis = if t then 1 - axis else axis in
                     let msg r what =
                       Printf.sprintf "%s of %s [%s] along %d (%s)" r.name what
                         (String.concat "x"
                            (List.map string_of_int (Array.to_list shape)))
                         axis layout
                     in
                     let input f =
                       let (Nx.P x) = lay (values dt stored f) layout in
                       Nx.P x
                     in
                     List.iter
                       (fun (what, f) ->
                         let (Nx.P x) = input f in
                         List.iter
                           (fun r ->
                             if r.ordered then
                               runs_as_on_host ~msg:(msg r what) r axis x)
                           runnings)
                       patterns;
                     if name <> "bool" then begin
                       let (Nx.P x) = input (List.assoc "ties" patterns) in
                       runs_as_on_host ~msg:(msg cumsum "ties") cumsum axis x;
                       let (Nx.P x) = input factors in
                       runs_as_on_host ~msg:(msg cumprod "factors") cumprod axis
                         x
                     end)
                   [ "stored"; "transposed" ])
               scan_geometries))
       served)

let scan_cases =
  group "scan cases"
    [
      test "running sums of -0 are +0, at every float dtype and length"
        (fun () ->
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
                    (Nx.P (Nx.zeros d [| 2; n |]))
                    (Nx.P (host (Nx.cumsum ~axis:1 x))))
                [ 1; 5; 300; 5000; 100_000 ])
            float_dtypes);
      test "a scan along an empty axis is empty" (fun () ->
          List.iter
            (fun shape ->
              let x = on_gpu (Nx.ones Nx.float32 shape) in
              equal (array int) shape (Nx.shape (host (Nx.cumsum ~axis:1 x))))
            [ [| 3; 0 |]; [| 0; 5 |] ]);
    ]

(* A running float sum of j terms lies within j eps sum |x| of the exact one. *)
let scan_bounds =
  group "scan bounds"
    [
      prop ~count:100
        "a float running sum lies within j eps sum |x| of the exact sum of its \
         j terms, plus a rounding to a narrow dtype"
        Gen.(
          quad
            (of_list ~pp:pp_dtype float_dtypes)
            (int_range 1 4) (int_range 0 20_000) int)
        (fun (Dtype d, rows, n, seed) ->
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
            Nx.to_array
              (Nx.cast Nx.float64 (host (Nx.cumsum ~axis:1 (on_gpu x))))
          in
          cover "parts" (rows = 1 && n > 4096);
          for r = 0 to rows - 1 do
            (* The prefixes' exact sums by one running compensated sum, and the
               element furthest past its bound. *)
            let sum = ref 0. and c = ref 0. and mag = ref 0. in
            let worst = ref (Float.neg_infinity, 0, 0., 0., 0.) in
            for j = 0 to n - 1 do
              let x = terms.((r * n) + j) in
              let t = !sum +. x in
              c :=
                !c
                +.
                if Float.abs !sum >= Float.abs x then !sum -. t +. x
                else x -. t +. !sum;
              sum := t;
              mag := !mag +. Float.abs x;
              let s = !sum +. !c in
              let e = Float.of_int (j + 1) *. f.eps *. !mag in
              let narrow = if f.m < 23 then ulp f (Float.abs s +. e) else 0. in
              let bound = e +. narrow +. (Float.epsilon *. Float.abs s) in
              let off = Float.abs (got.((r * n) + j) -. s) in
              let w, _, _, _, _ = !worst in
              if off -. bound > w then worst := (off -. bound, j, s, off, bound)
            done;
            let _, j, s, off, bound = !worst in
            if n > 0 then
              at_most
                ~msg:(Printf.sprintf "row %d, running sum %d, exact %h" r j s)
                float_exact ~than:bound off
          done);
    ]

(* Sorts *)

(* [x] sorted along [axis] on the GPU, values and positions read back, against
   the host's, bit for bit. *)
let sorts_as_on_host ?msg ~descending axis x =
  let packed (v, i) = (Nx.P v, Nx.P i) in
  let on (v, i) = (Nx.P (host v), Nx.P (host i)) in
  equal ?msg
    (pair Stored.packed Stored.packed)
    (packed (Nx.sort ~descending ~axis x))
    (on (Nx.sort ~descending ~axis (on_gpu x)))

let sorted =
  group "sorted"
    (List.map
       (fun (Stored.Case c) ->
         prop ~count:300
           (c.name
          ^ " values of every layout sort along an axis as on the host, both \
             ways")
           (Gen.pair (with_axis c.tensors) Gen.bool)
           (fun ((x, axis), descending) ->
             cover "strided" (not (Nx.is_c_contiguous x));
             cover "descending" descending;
             cover "an empty axis" (Nx.dim axis x = 0);
             cover "a one-element axis" (Nx.dim axis x = 1);
             sorts_as_on_host ~descending axis x))
       (List.filter served_case Stored.every))

(* Shapes and the axis they sort along, crossing the network's geometry: rows
   that share a block, rows of a block, and rows whose long steps run over all
   slots. *)
let sort_geometries =
  [
    ([| 70_000; 3 |], 1);
    ([| 1000; 300 |], 1);
    ([| 300; 1000 |], 0);
    ([| 3; 5000 |], 1);
    ([| 5000; 3 |], 0);
    ([| 1; 100_000 |], 1);
  ]

let sort_geometry =
  group "sort geometry"
    (List.map
       (fun (Dtype d as dt) ->
         test
           (Nx_dtype.to_string d
          ^ " rows of every length sort as on the host, both ways")
           (fun () ->
             List.iter
               (fun (shape, axis) ->
                 List.iter
                   (fun t ->
                     let stored =
                       if t then [| shape.(1); shape.(0) |] else shape
                     and axis = if t then 1 - axis else axis in
                     List.iter
                       (fun (what, f) ->
                         let (Nx.P x) = values dt stored f in
                         let x = if t then Nx.transpose x else x in
                         List.iter
                           (fun descending ->
                             let msg =
                               Printf.sprintf "%s [%s] along %d%s%s" what
                                 (String.concat "x"
                                    (List.map string_of_int
                                       (Array.to_list shape)))
                                 axis
                                 (if t then ", transposed" else "")
                                 (if descending then ", descending" else "")
                             in
                             sorts_as_on_host ~msg ~descending axis x)
                           [ false; true ])
                       patterns)
                   [ false; true ])
               sort_geometries))
       served)

let sort_cases =
  group "sort cases"
    [
      test "NaN sorts last, -0 below +0, descending the exact reverse"
        (fun () ->
          let x =
            on_gpu (Nx.create Nx.float32 [| 4 |] [| 1.; Float.nan; -0.; 0. |])
          in
          let bits xs = Array.map Int32.bits_of_float xs in
          let got descending =
            bits (Nx.to_array (host (fst (Nx.sort ~descending x))))
          in
          equal (array int32) (bits [| -0.; 0.; 1.; Float.nan |]) (got false);
          equal (array int32) (bits [| Float.nan; 1.; 0.; -0. |]) (got true));
      test "equal elements keep their order, both ways" (fun () ->
          let x =
            on_gpu (Nx.create Nx.int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |])
          in
          let got descending = Nx.to_array (host (Nx.argsort ~descending x)) in
          equal (array int64) [| 1L; 3L; 0L; 2L; 4L |] (got false);
          equal (array int64) [| 4L; 2L; 0L; 1L; 3L |] (got true));
    ]

(* Scatters *)

let modes = [ `Set; `Add; `Max; `Min ]

let mode_name = function
  | `Set -> "set"
  | `Add -> "add"
  | `Max -> "max"
  | `Min -> "min"

(* [x] scattered on the GPU, read back, against the host's: bit for bit, but for
   a NaN a float sum makes, which is any NaN. *)
let scatters_as_on_host ?msg ?(unique = false) mode ~axis ~indices ~values x =
  let f x indices values =
    Nx.scatter ~mode ~unique_indices:unique ~axis ~indices ~values x
  in
  let float = Nx_dtype.is_float (Nx.dtype x) in
  equal ?msg
    (if mode = `Add && float then floats_or_bits else Stored.packed)
    (Nx.P (f x indices values))
    (Nx.P (host (f (on_gpu x) (on_gpu indices) (on_gpu values))))

(* Whether no two indices of a row along [axis] are equal. *)
let distinct axis indices =
  let k = Nx.dim axis indices in
  let rows = Nx.to_array (Nx.moveaxis axis (Nx.ndim indices - 1) indices) in
  let ok = ref true in
  Array.iteri
    (fun i v ->
      for j = i - (i mod Int.max k 1) to i - 1 do
        if rows.(j) = v then ok := false
      done)
    rows;
  !ok

let scattered =
  group "scattered"
    (List.map
       (fun (Stored.Case c) ->
         prop ~count:300
           (c.name
          ^ " values of every layout scatter under each mode as on the host, \
             duplicate and out-of-range indices included")
           (Gen.triple
              (Gen.bind (with_axis c.tensors) (fun (x, axis) ->
                   let shape = Array.copy (Nx.shape x) in
                   let open Gen in
                   let* k = int_range 0 4 in
                   shape.(axis) <- k;
                   let+ ix =
                     array
                       ~size:(constant (Array.fold_left ( * ) 1 shape))
                       (positions (Nx.dim axis x))
                   in
                   (x, axis, Nx.create Nx.int64 shape ix)))
              (Gen.of_list
                 ~pp:(fun ppf m -> Format.pp_print_string ppf (mode_name m))
                 modes)
              Gen.bool)
           (fun ((x, axis, indices), mode, unique) ->
             (* Updates of the indices' shape, from [x]'s values. *)
             let values = Nx.flip (Nx.take_along_axis ~axis ~indices x) in
             let unique = unique && distinct axis indices in
             let float = Nx_dtype.is_float (Nx.dtype x) in
             List.iter (fun m -> cover (mode_name m) (m = mode)) modes;
             cover "strided" (not (Nx.is_c_contiguous x));
             cover "duplicates" (not (distinct axis indices));
             cover "unique" unique;
             (* A float sum's association is the GPU's own: with duplicates it
                differs from the host's, so it is compared where each position
                takes one update. *)
             if not (mode = `Add && float && not (distinct axis indices)) then
               scatters_as_on_host ~unique mode ~axis ~indices ~values x))
       (List.filter served_case Stored.every))

let scatter_cases =
  let f32 xs = Nx.create Nx.float32 [| Array.length xs |] xs in
  let i64 xs = Nx.create Nx.int64 [| Array.length xs |] xs in
  let bits t = Array.map Int32.bits_of_float (Nx.to_array t) in
  let on mode ?(unique = false) x indices values =
    host
      (Nx.scatter ~mode ~unique_indices:unique ~axis:0 ~indices:(on_gpu indices)
         ~values:(on_gpu values) (on_gpu x))
  in
  let nan_a = Int32.float_of_bits 0x7fc00001l
  and nan_b = Int32.float_of_bits 0x7fc00002l
  and nan_c = Int32.float_of_bits 0xffc00003l in
  group "scatter cases"
    [
      test "the last update to a position wins, in row-major order" (fun () ->
          equal (array float_exact) [| 4.; 0.; 3. |]
            (Nx.to_array
               (on `Set
                  (f32 [| 0.; 0.; 0. |])
                  (i64 [| 0L; 0L; 2L; 0L |])
                  (f32 [| 1.; 2.; 3.; 4. |]))));
      test "an update outside the axis is dropped, under every mode" (fun () ->
          List.iter
            (fun mode ->
              equal ~msg:(mode_name mode) (array float_exact) [| 5.; 7.; 5. |]
                (Nx.to_array
                   (on mode
                      (f32 [| 5.; 5.; 5. |])
                      (i64 [| -1L; 3L; Int64.min_int; Int64.max_int; 1L |])
                      (f32 [| 9.; 9.; 9.; 9.; 7. |]))))
            [ `Set; `Max ];
          equal ~msg:"add" (array float_exact) [| 5.; 12.; 5. |]
            (Nx.to_array
               (on `Add
                  (f32 [| 5.; 5.; 5. |])
                  (i64 [| -1L; 3L; Int64.min_int; Int64.max_int; 1L |])
                  (f32 [| 9.; 9.; 9.; 9.; 7. |]))));
      test "an extreme keeps the element's NaN, else the first NaN update's"
        (fun () ->
          let ix = i64 [| 0L; 0L; 1L; 1L |] in
          let ups = f32 [| nan_a; nan_b; nan_a; nan_b |] in
          List.iter
            (fun mode ->
              equal ~msg:(mode_name mode) (array int32)
                (bits (f32 [| nan_a; nan_c |]))
                (bits (on mode (f32 [| 1.; nan_c |]) ix ups)))
            [ `Max; `Min ]);
      test "an extreme orders -0 below +0" (fun () ->
          equal (array int32)
            (bits (f32 [| 0.; -0. |]))
            (bits
               (on `Max
                  (f32 [| -0.; -0. |])
                  (i64 [| 0L; 1L |])
                  (f32 [| 0.; -0. |])));
          equal (array int32)
            (bits (f32 [| -0.; 0. |]))
            (bits
               (on `Min
                  (f32 [| 0.; 0. |])
                  (i64 [| 0L; 1L |])
                  (f32 [| -0.; 0. |]))));
      test
        "a sum is +0 plus the element and its updates, untouched elements kept"
        (fun () ->
          equal (array int32)
            (bits (f32 [| 0.; 0.; -0. |]))
            (bits
               (on `Add
                  (f32 [| 1.; -0.; -0. |])
                  (i64 [| 0L; 1L |])
                  (f32 [| -1.; -0. |]))));
      cases
        ~name:(fun (Dtype d, _, _) -> Nx_dtype.to_string d)
        "a narrow float's updates sum at float32 and round once"
        [
          (Dtype Nx.float16, 2048., 2052.);
          (Dtype Nx.bfloat16, 256., 260.);
          (Dtype Nx.float8_e4m3, 16., 20.);
          (Dtype Nx.float8_e5m2, 16., 20.);
        ]
        (fun ((Dtype d as dt), big, expected) ->
          let (Nx.P x) = values dt [| 1 |] (fun _ -> big) in
          let ones = Nx.ones (Nx.dtype x) [| 3 |] in
          let got =
            host
              (Nx.scatter ~mode:`Add ~axis:0
                 ~indices:(on_gpu (Nx.zeros Nx.int64 [| 3 |]))
                 ~values:(on_gpu ones) (on_gpu x))
          in
          equal Stored.packed
            (Nx.P (Nx.cast d (Nx.create Nx.float64 [| 1 |] [| expected |])))
            (Nx.P got));
      test "many updates to few positions combine as on the host, every mode"
        (fun () ->
          let n = 1 lsl 20 in
          let indices =
            Nx.create Nx.int64 [| n |]
              (Array.init n (fun i -> Int64.of_int (i * 7919 mod 3)))
          in
          List.iter
            (fun (Dtype d as dt) ->
              let (Nx.P x) = values dt [| 3 |] (fun i -> Float.of_int i) in
              let (Nx.P v) =
                values dt [| n |] (fun i -> Float.of_int (spread i - 6))
              in
              let v = Nx.unpack (Nx.dtype x) (Nx.P v) in
              List.iter
                (fun mode ->
                  scatters_as_on_host
                    ~msg:(Nx_dtype.to_string d ^ " " ^ mode_name mode)
                    mode ~axis:0 ~indices ~values:v x)
                modes)
            served);
    ]

(* Windows *)

(* A window geometry over [k] axes. *)
type geometry = {
  kernel_size : int array;
  stride : int array;
  dilation : int array;
  padding : (int * int) array;
}

let pp_geometry ppf g =
  let ints a = String.concat "," (List.map string_of_int (Array.to_list a)) in
  Format.fprintf ppf "kernel %s stride %s dilation %s padding %s"
    (ints g.kernel_size) (ints g.stride) (ints g.dilation)
    (String.concat ","
       (List.map
          (fun (a, b) -> Printf.sprintf "%d/%d" a b)
          (Array.to_list g.padding)))

let window_geometry k =
  Gen.with_pp pp_geometry
    Gen.(
      let a lo hi = array ~size:(constant k) (int_range lo hi) in
      let+ kernel_size = a 1 3
      and+ stride = a 1 3
      and+ dilation = a 1 2
      and+ padding =
        array ~size:(constant k) (pair (int_range 0 2) (int_range 0 2))
      in
      { kernel_size; stride; dilation; padding })

let patches g x =
  Nx.extract_patches ~kernel_size:g.kernel_size ~stride:g.stride
    ~dilation:g.dilation ~padding:g.padding x

let combined g output_size p =
  Nx.combine_patches ~output_size ~kernel_size:g.kernel_size ~stride:g.stride
    ~dilation:g.dilation ~padding:g.padding p

(* [x]'s windows under [g] on the GPU, and their sum back, read back, against
   the host's: the windows bit for bit, the sums too but for a NaN a float sum
   makes, which is any NaN. *)
let windows_as_on_host ?msg g x =
  let k = Array.length g.kernel_size in
  let r = Nx.ndim x in
  let output_size = Array.sub (Nx.shape x) (r - k) k in
  let p = patches g x and gp = patches g (on_gpu x) in
  equal ?msg Stored.packed (Nx.P p) (Nx.P (host gp));
  let float = Nx_dtype.is_float (Nx.dtype x) in
  equal ?msg
    (if float then floats_or_bits else Stored.packed)
    (Nx.P (combined g output_size p))
    (Nx.P (host (combined g output_size gp)))

let windowed =
  group "windowed"
    (List.map
       (fun (Stored.Case c) ->
         prop ~count:300
           (c.name
          ^ " values of every layout unfold and fold as on the host, under \
             every geometry")
           (Gen.bind c.tensors (fun x ->
                let x = if Nx.ndim x = 0 then Nx.reshape [| 1 |] x else x in
                let open Gen in
                let* k = int_range 1 (Int.min 2 (Nx.ndim x)) in
                let+ g = window_geometry k in
                (x, g)))
           (fun (x, g) ->
             let k = Array.length g.kernel_size in
             cover "strided" (not (Nx.is_c_contiguous x));
             cover "two axes" (k = 2);
             cover "padded" (Array.exists (fun (a, b) -> a + b > 0) g.padding);
             let spatial = Array.sub (Nx.shape x) (Nx.ndim x - k) k in
             cover "no window"
               (List.exists
                  (fun d ->
                    let a, b = g.padding.(d) in
                    spatial.(d) + a + b
                    < (g.dilation.(d) * (g.kernel_size.(d) - 1)) + 1)
                  (List.init k Fun.id));
             windows_as_on_host g x))
       (List.filter served_case Stored.every))

let window_cases =
  let g k s d p = { kernel_size = k; stride = s; dilation = d; padding = p } in
  group "window cases"
    [
      test "overlapping windows sum where they overlap" (fun () ->
          let g = g [| 2 |] [| 1 |] [| 1 |] [| (0, 0) |] in
          let p = patches g (on_gpu (Nx.ones Nx.float32 [| 1; 4 |])) in
          equal (array float_exact) [| 1.; 2.; 2.; 1. |]
            (Nx.to_array (host (combined g [| 4 |] p))));
      test "a tap in the padding reads 0" (fun () ->
          let g = g [| 3 |] [| 1 |] [| 1 |] [| (1, 1) |] in
          let x = on_gpu (Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |]) in
          equal (array float_exact)
            [| 0.; 1.; 2.; 1.; 2.; 3.; 2.; 3.; 0. |]
            (Nx.to_array (host (patches g x))));
      test "large windows of every dtype unfold and fold as on the host"
        (fun () ->
          List.iter
            (fun (Dtype d as dt) ->
              List.iter
                (fun (shape, gm) ->
                  let (Nx.P x) =
                    values dt shape (fun i -> Float.of_int (spread i - 6))
                  in
                  windows_as_on_host
                    ~msg:
                      (Format.asprintf "%s %a" (Nx_dtype.to_string d)
                         pp_geometry gm)
                    gm x;
                  windows_as_on_host
                    ~msg:
                      (Format.asprintf "%s transposed %a" (Nx_dtype.to_string d)
                         pp_geometry gm)
                    gm (Nx.transpose x))
                [
                  ( [| 2; 3; 64; 65 |],
                    g [| 3; 3 |] [| 1; 1 |] [| 1; 1 |] [| (1, 1); (1, 1) |] );
                  ( [| 2; 3; 64; 65 |],
                    g [| 3; 2 |] [| 2; 3 |] [| 2; 1 |] [| (0, 2); (1, 0) |] );
                  ([| 1; 100_000 |], g [| 5 |] [| 1 |] [| 1 |] [| (2, 2) |]);
                ])
            served);
    ]

(* Random values *)

let key_words (k : Nx.Rng.t) = Nx.P (host (k :> Nx.int32_t))
let gpu_key seed = Nx.Rng.of_tensor (on_gpu (Nx.Rng.key seed :> Nx.int32_t))

let random =
  group "random"
    [
      test "an RNG key on the GPU splits as on the host" (fun () ->
          equal (array Stored.packed)
            (Array.map key_words (Nx.Rng.split (Nx.Rng.key 42)))
            (Array.map key_words (Nx.Rng.split (gpu_key 42))));
      prop "keys on the GPU split, fold in and draw as on the host"
        Gen.(triple int (int_range 1 5) (pair int (int_range 0 70)))
        (fun (seed, n, (data, size)) ->
          let here = Nx.Rng.key seed and there = gpu_key seed in
          equal ~msg:"split" (array Stored.packed)
            (Array.map key_words (Nx.Rng.split ~n here))
            (Array.map key_words (Nx.Rng.split ~n there));
          equal ~msg:"split_batch" Stored.packed
            (key_words (Nx.Rng.split_batch ~n here))
            (key_words (Nx.Rng.split_batch ~n there));
          equal ~msg:"fold_in" Stored.packed
            (key_words (Nx.Rng.fold_in here data))
            (key_words (Nx.Rng.fold_in there data));
          equal ~msg:"uniform" Stored.packed
            (Nx.P (Nx.Rng.uniform here Nx.float32 [| size |]))
            (Nx.P (host (Nx.Rng.uniform there Nx.float32 [| size |]))));
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
         moves;
         move_cases;
         scanned;
         scan_geometry;
         scan_cases;
         scan_bounds;
         random;
         sorted;
         sort_geometry;
         sort_cases;
         scattered;
         scatter_cases;
         windowed;
         window_cases;
         accuracy;
       ])
