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
            (Exn.invalid_arg
               ~substring:"no reductions. Compile it with Rune.jit") (fun () ->
              ignore (Nx.sum x)));
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

let accuracy = group "accuracy" (Nx_test.Accuracy.groups { put = on_gpu })

let () =
  exit
    (run "nx.amd"
       [
         domains; opening; computing; conformance; elementwise; sweep; accuracy;
       ])
