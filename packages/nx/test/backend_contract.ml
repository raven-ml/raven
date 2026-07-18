(* Maintenance contract for Nx backends.

   [Make (B) (C)] generates a Windtrap suite that gates a backend against the
   public backend semantics. It is written against [Nx_core.Backend_intf.S]
   alone so any backend can instantiate it; correctness oracles are pure OCaml,
   computed independently of the backend under test. *)

module Make
    (B : Nx_core.Backend_intf.S)
    (C : sig
      val create_context : unit -> B.context
    end) =
struct
  open Windtrap
  module F = Nx_core.Make_frontend (B)
  module Dtype = Nx_core.Dtype
  module View = Nx_core.View

  let ctx = C.create_context ()

  type expectation = Pass

  (* ───── Numeric helpers ───── *)

  let prod a = Array.fold_left ( * ) 1 a

  let row_major_strides shape =
    let n = Array.length shape in
    let s = Array.make n 1 in
    for i = n - 2 downto 0 do
      s.(i) <- s.(i + 1) * shape.(i + 1)
    done;
    s

  let unravel lin shape =
    let n = Array.length shape in
    let idx = Array.make n 0 in
    let r = ref lin in
    for i = n - 1 downto 0 do
      idx.(i) <- !r mod shape.(i);
      r := !r / shape.(i)
    done;
    idx

  (* Row-major data of [permute shape axes] applied to row-major [a]. Also
     serves for 2-D transpose ([axes = [|1;0|]]). *)
  let permute_arr shape axes (a : 'a array) : 'a array =
    let nd = Array.length shape in
    let out_shape = Array.map (fun ax -> shape.(ax)) axes in
    let istr = row_major_strides shape in
    let n = prod shape in
    if n = 0 then [||]
    else begin
      let out = Array.make n a.(0) in
      for oi = 0 to n - 1 do
        let idx = unravel oi out_shape in
        let ii = ref 0 in
        for k = 0 to nd - 1 do
          ii := !ii + (idx.(k) * istr.(axes.(k)))
        done;
        out.(oi) <- a.(!ii)
      done;
      out
    end

  let reverse_last rows cols (a : 'a array) : 'a array =
    Array.init (rows * cols) (fun i ->
        let r = i / cols and c = i mod cols in
        a.((r * cols) + (cols - 1 - c)))

  let take_cols rows total c0 c1 (a : 'a array) : 'a array =
    let w = c1 - c0 in
    Array.init (rows * w) (fun i ->
        let r = i / w and c = i mod w in
        a.((r * total) + c0 + c))

  let tile_rows rows (row : 'a array) : 'a array =
    let w = Array.length row in
    Array.init (rows * w) (fun i -> row.(i mod w))

  (* ───── Float comparison (NaN- and inf-aware) ───── *)

  let float_close ~rel ~abs a b =
    if Float.is_nan a && Float.is_nan b then true
    else if Float.is_nan a || Float.is_nan b then false
    else if a = b then true (* exact, and handles matching infinities *)
    else
      let d = Float.abs (a -. b) in
      d <= abs || d <= rel *. Float.max (Float.abs a) (Float.abs b)

  let ftest ~rel ~abs : float testable =
    testable
      ~pp:(fun ppf x -> Format.fprintf ppf "%.9g" x)
      ~equal:(float_close ~rel ~abs) ()

  let ctest ~rel ~abs : Complex.t testable =
    testable
      ~pp:(fun ppf z ->
        Format.fprintf ppf "(%.9g, %.9g)" z.Complex.re z.Complex.im)
      ~equal:(fun a b ->
        float_close ~rel ~abs a.Complex.re b.Complex.re
        && float_close ~rel ~abs a.Complex.im b.Complex.im)
      ()

  (* ───── Dtype tables ─────

     Tolerances are dtype- and op-class-aware. [a_*] are for arithmetic /
     algebraic ops whose only error is the final round to the dtype; [t_*] are
     for transcendental / iterative ops that additionally carry library
     approximation error. Values are chosen so a correct kernel passes with
     margin and a wrong result (an off-by-one ulp bug excepted) fails. *)

  type fdt =
    | FDT : {
        dt : (float, 'b) Dtype.t;
        name : string;
        pool : float array;
        a_rel : float;
        a_abs : float;
        t_rel : float;
        t_abs : float;
      }
        -> fdt

  let fpool =
    [|
      0.5;
      -1.5;
      2.0;
      -0.25;
      3.0;
      1.0;
      -2.0;
      0.75;
      -3.5;
      4.0;
      -0.5;
      2.5;
      1.25;
      -1.0;
      0.0;
      5.0;
      -4.0;
      1.75;
    |]

  let float_dtypes =
    [
      FDT
        {
          dt = F.float64;
          name = "f64";
          pool = fpool;
          a_rel = 1e-11;
          a_abs = 1e-12;
          t_rel = 1e-9;
          t_abs = 1e-9;
        };
      FDT
        {
          dt = F.float32;
          name = "f32";
          pool = fpool;
          a_rel = 1e-5;
          a_abs = 1e-6;
          t_rel = 1e-4;
          t_abs = 1e-5;
        };
      FDT
        {
          dt = F.float16;
          name = "f16";
          pool = fpool;
          a_rel = 5e-3;
          a_abs = 5e-3;
          t_rel = 2e-2;
          t_abs = 2e-2;
        };
      FDT
        {
          dt = F.bfloat16;
          name = "bf16";
          pool = fpool;
          a_rel = 3e-2;
          a_abs = 3e-2;
          t_rel = 6e-2;
          t_abs = 6e-2;
        };
    ]

  (* fp8 has too few mantissa bits for the generic elementwise oracle to be
     discriminating; it is exercised only where the contract requires (cast). *)

  let f_arith (FDT d) = ftest ~rel:d.a_rel ~abs:d.a_abs
  let f_trans (FDT d) = ftest ~rel:d.t_rel ~abs:d.t_abs

  type idt =
    | IDT : {
        dt : ('a, 'b) Dtype.t;
        name : string;
        bits : int;
        signed : bool;
        of_i64 : int64 -> 'a;
        to_i64 : 'a -> int64; (* sign-extended for signed carriers *)
        pool : 'a array;
        tst : 'a testable;
      }
        -> idt

  let ipool_signed =
    [| 3; -7; 1; -10; 5; 2; -8; 4; 9; -6; 0; 12; -11; 15; -13; 14; -20; 17 |]

  (* Unsigned pools carry high-bit-set values so unsigned and signed
     interpretations diverge — otherwise div/mod/max/min/sort/compare give the
     same answer either way and a backend that used signed ops for an unsigned
     dtype would pass. u8/u16 are hardware-unsigned bigarrays; 128..255 sets
     u8's high bit. u32/u64 carry software unsigned semantics over int32/int64,
     the sharp case, so their pools set bit 31 / bit 63. No value is -1, so no
     signed min_int/-1 division overflow is reachable. *)
  let ipool_unsigned =
    [| 3; 200; 1; 250; 5; 2; 130; 4; 9; 6; 0; 255; 11; 128; 13; 14; 240; 17 |]

  let i32pool_s = Array.map Int32.of_int ipool_signed

  let i32pool_u =
    [|
      3l;
      0x80000000l;
      1l;
      0xFFFFFFF0l;
      5l;
      2l;
      0xC0000000l;
      4l;
      9l;
      6l;
      0l;
      0x90000000l;
      11l;
      0xA5A5A5A5l;
      13l;
      14l;
      0xF0000000l;
      17l;
    |]

  let i64pool_s = Array.map Int64.of_int ipool_signed

  let i64pool_u =
    [|
      3L;
      0x8000000000000000L;
      1L;
      0xFFFFFFFFFFFFFFF0L;
      5L;
      2L;
      0xC000000000000000L;
      4L;
      9L;
      6L;
      0L;
      0x9000000000000000L;
      11L;
      0xA5A5A5A5A5A5A5A5L;
      13L;
      14L;
      0xF000000000000000L;
      17L;
    |]

  (* u16 needs its own pool: [ipool_unsigned] is capped at 255 for u8, so bit 15
     (>= 32768) is never set there and u16's unsigned semantics go
     unstressed. *)
  let u16pool =
    [|
      3;
      40000;
      1;
      250;
      5;
      2;
      32768;
      4;
      9;
      6;
      0;
      60000;
      11;
      128;
      13;
      14;
      50000;
      17;
    |]

  let int_dtypes =
    [
      IDT
        {
          dt = F.int8;
          name = "i8";
          bits = 8;
          signed = true;
          of_i64 = Int64.to_int;
          to_i64 = Int64.of_int;
          pool = ipool_signed;
          tst = int;
        };
      IDT
        {
          dt = F.uint8;
          name = "u8";
          bits = 8;
          signed = false;
          of_i64 = Int64.to_int;
          to_i64 = Int64.of_int;
          pool = ipool_unsigned;
          tst = int;
        };
      IDT
        {
          dt = F.int16;
          name = "i16";
          bits = 16;
          signed = true;
          of_i64 = Int64.to_int;
          to_i64 = Int64.of_int;
          pool = ipool_signed;
          tst = int;
        };
      IDT
        {
          dt = F.uint16;
          name = "u16";
          bits = 16;
          signed = false;
          of_i64 = Int64.to_int;
          to_i64 = Int64.of_int;
          pool = u16pool;
          tst = int;
        };
      IDT
        {
          dt = F.int32;
          name = "i32";
          bits = 32;
          signed = true;
          of_i64 = Int64.to_int32;
          to_i64 = Int64.of_int32;
          pool = i32pool_s;
          tst = int32;
        };
      IDT
        {
          dt = F.uint32;
          name = "u32";
          bits = 32;
          signed = false;
          of_i64 = Int64.to_int32;
          to_i64 = Int64.of_int32;
          pool = i32pool_u;
          tst = int32;
        };
      IDT
        {
          dt = F.int64;
          name = "i64";
          bits = 64;
          signed = true;
          of_i64 = Fun.id;
          to_i64 = Fun.id;
          pool = i64pool_s;
          tst = int64;
        };
      IDT
        {
          dt = F.uint64;
          name = "u64";
          bits = 64;
          signed = false;
          of_i64 = Fun.id;
          to_i64 = Fun.id;
          pool = i64pool_u;
          tst = int64;
        };
    ]

  (* Exact integer oracle in int64, with dtype-width wrap. add/sub/mul are
     bit-identical for signed and unsigned two's complement, so plain Int64
     arithmetic + wrap is correct for both; div/mod/compare branch on
     signedness. *)

  let wrap ~bits ~signed (x : int64) : int64 =
    if bits >= 64 then x
    else
      let m = Int64.sub (Int64.shift_left 1L bits) 1L in
      let v = Int64.logand x m in
      if signed && Int64.logand v (Int64.shift_left 1L (bits - 1)) <> 0L then
        Int64.sub v (Int64.shift_left 1L bits)
      else v

  (* Unsigned magnitude of a stored value, as a non-negative int64 for bits<64,
     or the raw pattern for bits=64 (then compare/divide with the unsigned Int64
     operations). *)
  let umag ~bits (x : int64) : int64 =
    if bits >= 64 then x
    else Int64.logand x (Int64.sub (Int64.shift_left 1L bits) 1L)

  (* ───── Layout matrix ─────

     For a dtype and an element pool, produce a set of tensors whose logical
     content is known, each exercising a different view: contiguous, transposed
     (permute), offset slice (shrink), broadcast (expand, 0 strides), reversed
     (flip, negative strides), a permuted 3-D case (multi-axis strides), a
     contiguous rank-5 case, a scalar (rank 0), and an empty (size-0) case.

     Logical arrays are computed in OCaml from the STORED base values (read once
     from the contiguous base, so low-precision rounding is reflected in the
     oracle), and every view is derived from a base via movement ops, so the
     only backend primitives trusted to build the oracle input are contiguous
     construction and readback — the same ones used to read results. *)

  let layouts (type a b) (dt : (a, b) Dtype.t) (pool : a array) :
      (string * (a, b) B.t * a array * int array) list =
    let mk shape data = F.create ctx dt shape data in
    let stored t = F.to_array t in
    let sub n = Array.sub pool 0 n in
    let d12 = sub 12 in
    let contig = mk [| 3; 4 |] d12 in
    let s_contig = stored contig in
    let b43 = mk [| 4; 3 |] d12 in
    let s43 = stored b43 in
    let b36 = mk [| 3; 6 |] (sub 18) in
    let s36 = stored b36 in
    let b14 = mk [| 1; 4 |] (sub 4) in
    let s14 = stored b14 in
    let b232 = mk [| 2; 3; 2 |] d12 in
    let s232 = stored b232 in
    let hr5 = mk [| 1; 3; 1; 2; 2 |] d12 in
    let s_hr5 = stored hr5 in
    let scal = mk [||] (sub 1) in
    let empty = mk [| 0; 4 |] [||] in
    [
      ("contig", contig, s_contig, [| 3; 4 |]);
      ( "transpose",
        B.permute b43 [| 1; 0 |],
        permute_arr [| 4; 3 |] [| 1; 0 |] s43,
        [| 3; 4 |] );
      ( "slice",
        B.shrink b36 [| (0, 3); (1, 5) |],
        take_cols 3 6 1 5 s36,
        [| 3; 4 |] );
      (* broadcast = stride-0 expand. This contract guards values through the
         abstract backend interface; backend-local ABI tests separately pin
         whether an implementation consumes the view without materialization. *)
      ("broadcast", B.expand b14 [| 3; 4 |], tile_rows 3 s14, [| 3; 4 |]);
      ( "flip",
        B.flip contig [| false; true |],
        reverse_last 3 4 s_contig,
        [| 3; 4 |] );
      ( "permute3",
        B.permute b232 [| 2; 0; 1 |],
        permute_arr [| 2; 3; 2 |] [| 2; 0; 1 |] s232,
        [| 2; 2; 3 |] );
      ("rank5", hr5, s_hr5, [| 1; 3; 1; 2; 2 |]);
      ("scalar", scal, stored scal, [||]);
      ("empty", empty, [||], [| 0; 4 |]);
    ]

  (* ───── Polymorphic op wrappers ─────

     A function argument in OCaml is monomorphic, but the drivers apply an op at
     many dtypes; these records restore the needed rank-2 polymorphism. *)

  type un = { u1 : 'a 'b. ('a, 'b) B.t -> ('a, 'b) B.t }
  type bin = { b2 : 'a 'b. ('a, 'b) B.t -> ('a, 'b) B.t -> ('a, 'b) B.t }

  type cmp = {
    c2 : 'a 'b. ('a, 'b) B.t -> ('a, 'b) B.t -> (bool, Dtype.bool_elt) B.t;
  }

  let case classify path name body =
    let full = if path = "" then name else path ^ "/" ^ name in
    match classify full with Pass -> test name body

  (* ───── Elementwise: unary float ───── *)

  let unary_float_group classify path name (op : un) ~cls ref_fn =
    let tsel = if cls = `Trans then f_trans else f_arith in
    group name
      (List.map
         (fun (FDT d as fdt) ->
           let tst = tsel fdt in
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   let expected = Array.map ref_fn logical in
                   let got = op.u1 t in
                   equal ~msg:("shape " ^ lname) (array int) shape (F.shape got);
                   equal ~msg:lname (array tst) expected (F.to_array got))
                 (layouts d.dt d.pool)))
         float_dtypes)

  let unary_float_tests classify path =
    [
      unary_float_group classify path "neg" { u1 = B.neg } ~cls:`Arith (fun x ->
          -.x);
      unary_float_group classify path "abs" { u1 = B.abs } ~cls:`Arith Float.abs;
      unary_float_group classify path "recip" { u1 = B.recip } ~cls:`Arith
        (fun x -> 1.0 /. x);
      unary_float_group classify path "sign" { u1 = B.sign } ~cls:`Arith
        (fun x ->
          if Float.is_nan x then Float.nan
          else if x > 0.0 then 1.0
          else if x < 0.0 then -1.0
          else 0.0);
      unary_float_group classify path "sqrt" { u1 = B.sqrt } ~cls:`Trans
        Float.sqrt;
      unary_float_group classify path "exp" { u1 = B.exp } ~cls:`Trans Float.exp;
      unary_float_group classify path "log" { u1 = B.log } ~cls:`Trans Float.log;
      unary_float_group classify path "sin" { u1 = B.sin } ~cls:`Trans Float.sin;
      unary_float_group classify path "cos" { u1 = B.cos } ~cls:`Trans Float.cos;
      unary_float_group classify path "tan" { u1 = B.tan } ~cls:`Trans Float.tan;
      unary_float_group classify path "asin" { u1 = B.asin } ~cls:`Trans
        Float.asin;
      unary_float_group classify path "acos" { u1 = B.acos } ~cls:`Trans
        Float.acos;
      unary_float_group classify path "atan" { u1 = B.atan } ~cls:`Trans
        Float.atan;
      unary_float_group classify path "sinh" { u1 = B.sinh } ~cls:`Trans
        Float.sinh;
      unary_float_group classify path "cosh" { u1 = B.cosh } ~cls:`Trans
        Float.cosh;
      unary_float_group classify path "tanh" { u1 = B.tanh } ~cls:`Trans
        Float.tanh;
      unary_float_group classify path "trunc" { u1 = B.trunc } ~cls:`Arith
        Float.trunc;
      unary_float_group classify path "ceil" { u1 = B.ceil } ~cls:`Arith
        Float.ceil;
      unary_float_group classify path "floor" { u1 = B.floor } ~cls:`Arith
        Float.floor;
      unary_float_group classify path "round" { u1 = B.round } ~cls:`Arith
        Float.round;
      unary_float_group classify path "erf" { u1 = B.erf } ~cls:`Trans Float.erf;
    ]

  (* ───── Elementwise: binary float ─────

     Both operands vary in layout independently, over shape [3;4], to exercise
     mixed-stride combinations feeding one kernel. *)

  let float_bin_cases (type b) (dt : (float, b) Dtype.t) poolA poolB =
    let mk shape data = F.create ctx dt shape data in
    let stored t = F.to_array t in
    (* B draws from a shifted slice so the two operands differ even when the two
       pools are the same, exercising order-sensitive ops (sub, div, atan2). *)
    let subA n = Array.sub poolA 0 n and subB n = Array.sub poolB 4 n in
    let cA = mk [| 3; 4 |] (subA 12) and cB = mk [| 3; 4 |] (subB 12) in
    let sA = stored cA and sB = stored cB in
    let tB43 = mk [| 4; 3 |] (subB 12) in
    let stB43 = stored tB43 in
    let bA14 = mk [| 1; 4 |] (subA 4) in
    let sbA14 = stored bA14 in
    let scalA = mk [||] (subA 1) and scalB = mk [||] (subB 1) in
    let r5A = mk [| 1; 3; 1; 2; 2 |] (subA 12)
    and r5B = mk [| 1; 3; 1; 2; 2 |] (subB 12) in
    [
      ("contig,contig", cA, cB, sA, sB);
      ( "contig,transpose",
        cA,
        B.permute tB43 [| 1; 0 |],
        sA,
        permute_arr [| 4; 3 |] [| 1; 0 |] stB43 );
      ("broadcast,contig", B.expand bA14 [| 3; 4 |], cB, tile_rows 3 sbA14, sB);
      ( "flip,slice",
        B.flip cA [| false; true |],
        B.shrink (mk [| 3; 6 |] (Array.sub poolB 0 18)) [| (0, 3); (1, 5) |],
        reverse_last 3 4 sA,
        take_cols 3 6 1 5 (stored (mk [| 3; 6 |] (Array.sub poolB 0 18))) );
      ("scalar,scalar", scalA, scalB, stored scalA, stored scalB);
      ("empty,empty", mk [| 0; 4 |] [||], mk [| 0; 4 |] [||], [||], [||]);
      ("rank5,rank5", r5A, r5B, stored r5A, stored r5B);
      ( "permute3,permute3",
        B.permute (mk [| 2; 3; 2 |] (subA 12)) [| 2; 0; 1 |],
        B.permute (mk [| 2; 3; 2 |] (subB 12)) [| 2; 0; 1 |],
        permute_arr [| 2; 3; 2 |] [| 2; 0; 1 |]
          (stored (mk [| 2; 3; 2 |] (subA 12))),
        permute_arr [| 2; 3; 2 |] [| 2; 0; 1 |]
          (stored (mk [| 2; 3; 2 |] (subB 12))) );
    ]

  let binary_float_group classify path name (op : bin) ~cls ref_fn =
    let tsel = if cls = `Trans then f_trans else f_arith in
    group name
      (List.map
         (fun (FDT d as fdt) ->
           let tst = tsel fdt in
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, ta, tb, la, lb) ->
                   let expected = Array.map2 ref_fn la lb in
                   let got = op.b2 ta tb in
                   equal ~msg:lname (array tst) expected (F.to_array got))
                 (float_bin_cases d.dt d.pool (Array.map (fun x -> x) d.pool))))
         float_dtypes)

  let binary_float_tests classify path =
    [
      binary_float_group classify path "add" { b2 = B.add } ~cls:`Arith ( +. );
      binary_float_group classify path "sub" { b2 = B.sub } ~cls:`Arith ( -. );
      binary_float_group classify path "mul" { b2 = B.mul } ~cls:`Arith ( *. );
      binary_float_group classify path "div" { b2 = B.fdiv } ~cls:`Arith ( /. );
      binary_float_group classify path "max" { b2 = B.max } ~cls:`Arith
        Float.max;
      binary_float_group classify path "min" { b2 = B.min } ~cls:`Arith
        Float.min;
      binary_float_group classify path "pow" { b2 = B.pow } ~cls:`Trans
        Float.pow;
      binary_float_group classify path "atan2" { b2 = B.atan2 } ~cls:`Trans
        Float.atan2;
    ]

  (* ───── Elementwise: integer arithmetic and bitwise ───── *)

  let int_bin_cases (type a b) (dt : (a, b) Dtype.t) (poolA : a array)
      (poolB : a array) =
    let mk shape data = F.create ctx dt shape data in
    let stored t = F.to_array t in
    let cA = mk [| 3; 4 |] (Array.sub poolA 0 12) in
    let cB = mk [| 3; 4 |] (Array.sub poolB 4 12) in
    let sA = stored cA and sB = stored cB in
    let tB43 = mk [| 4; 3 |] (Array.sub poolB 4 12) in
    let stB43 = stored tB43 in
    let scalA = mk [||] (Array.sub poolA 0 1)
    and scalB = mk [||] (Array.sub poolB 4 1) in
    [
      ("contig,contig", cA, cB, sA, sB);
      ( "contig,transpose",
        cA,
        B.permute tB43 [| 1; 0 |],
        sA,
        permute_arr [| 4; 3 |] [| 1; 0 |] stB43 );
      ("flip,contig", B.flip cA [| false; true |], cB, reverse_last 3 4 sA, sB);
      ("scalar,scalar", scalA, scalB, stored scalA, stored scalB);
      ("empty,empty", mk [| 0; 4 |] [||], mk [| 0; 4 |] [||], [||], [||]);
      ( "rank5,rank5",
        mk [| 1; 3; 1; 2; 2 |] (Array.sub poolA 0 12),
        mk [| 1; 3; 1; 2; 2 |] (Array.sub poolB 4 12),
        stored (mk [| 1; 3; 1; 2; 2 |] (Array.sub poolA 0 12)),
        stored (mk [| 1; 3; 1; 2; 2 |] (Array.sub poolB 4 12)) );
      ( "permute3,permute3",
        B.permute (mk [| 2; 3; 2 |] (Array.sub poolA 0 12)) [| 2; 0; 1 |],
        B.permute (mk [| 2; 3; 2 |] (Array.sub poolB 4 12)) [| 2; 0; 1 |],
        permute_arr [| 2; 3; 2 |] [| 2; 0; 1 |]
          (stored (mk [| 2; 3; 2 |] (Array.sub poolA 0 12))),
        permute_arr [| 2; 3; 2 |] [| 2; 0; 1 |]
          (stored (mk [| 2; 3; 2 |] (Array.sub poolB 4 12))) );
    ]

  (* [nonzero_b] guards div/mod: replace any zero in the divisor pool with one,
     so the general cases never divide by zero (that is a dedicated regression
     case). Dividend zeros are kept — 0 / x and 0 mod x are well-defined. *)
  let int_binary_group classify path name (op : bin) ~nonzero_b ref64 =
    group name
      (List.map
         (fun (IDT d) ->
           let poolB =
             if nonzero_b then
               Array.map
                 (fun x -> if d.to_i64 x = 0L then d.of_i64 1L else x)
                 d.pool
             else d.pool
           in
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, ta, tb, la, lb) ->
                   let expected =
                     Array.map2
                       (fun a b ->
                         d.of_i64
                           (wrap ~bits:d.bits ~signed:d.signed
                              (ref64 ~bits:d.bits ~signed:d.signed (d.to_i64 a)
                                 (d.to_i64 b))))
                       la lb
                   in
                   let got = op.b2 ta tb in
                   equal ~msg:lname (array d.tst) expected (F.to_array got))
                 (int_bin_cases d.dt d.pool poolB)))
         int_dtypes)

  (* raw int64 op semantics; wrap is applied by the caller *)
  let o_add ~bits:_ ~signed:_ a b = Int64.add a b
  let o_sub ~bits:_ ~signed:_ a b = Int64.sub a b
  let o_mul ~bits:_ ~signed:_ a b = Int64.mul a b

  let o_div ~bits ~signed a b =
    if b = 0L then 0L (* Contract: integer div by zero -> 0. *)
    else if signed then Int64.div a b
    else Int64.unsigned_div (umag ~bits a) (umag ~bits b)

  let o_mod ~bits ~signed a b =
    if b = 0L then 0L (* Contract: integer mod by zero -> 0. *)
    else if signed then Int64.rem a b
    else Int64.unsigned_rem (umag ~bits a) (umag ~bits b)

  let o_max ~bits ~signed a b =
    if signed then if Int64.compare a b >= 0 then a else b
    else if Int64.unsigned_compare (umag ~bits a) (umag ~bits b) >= 0 then a
    else b

  let o_min ~bits ~signed a b =
    if signed then if Int64.compare a b <= 0 then a else b
    else if Int64.unsigned_compare (umag ~bits a) (umag ~bits b) <= 0 then a
    else b

  let o_and ~bits:_ ~signed:_ a b = Int64.logand a b
  let o_or ~bits:_ ~signed:_ a b = Int64.logor a b
  let o_xor ~bits:_ ~signed:_ a b = Int64.logxor a b

  (* Integer pow (caller wraps to dtype width). Negative exponents: 1/base^|e|
     truncates toward zero, so the result is 0 unless |base| = 1 (base 1 -> 1;
     base -1 -> ±1 by parity). base^0 = 1. Non-negative exponents accumulate
     with wrapping multiply. *)
  let ipow64 base exp =
    if exp < 0L then
      if base = 1L then 1L
      else if base = -1L then if Int64.rem exp 2L = 0L then 1L else -1L
      else 0L
    else begin
      let r = ref 1L and e = ref exp in
      while !e > 0L do
        r := Int64.mul !r base;
        e := Int64.sub !e 1L
      done;
      !r
    end

  let o_neg ~bits ~signed x = wrap ~bits ~signed (Int64.neg x)

  (* abs: signed uses Int64.abs (abs INT_MIN = INT_MIN, then wrap keeps it);
     unsigned magnitudes are already non-negative, so abs is the identity. *)
  let o_abs ~bits ~signed x =
    if signed then wrap ~bits ~signed (Int64.abs x) else x

  let int_binary_tests classify path =
    [
      int_binary_group classify path "add" { b2 = B.add } ~nonzero_b:false o_add;
      int_binary_group classify path "sub" { b2 = B.sub } ~nonzero_b:false o_sub;
      int_binary_group classify path "mul" { b2 = B.mul } ~nonzero_b:false o_mul;
      int_binary_group classify path "div" { b2 = B.idiv } ~nonzero_b:true o_div;
      int_binary_group classify path "mod" { b2 = B.mod_ } ~nonzero_b:true o_mod;
      int_binary_group classify path "max" { b2 = B.max } ~nonzero_b:false o_max;
      int_binary_group classify path "min" { b2 = B.min } ~nonzero_b:false o_min;
      int_binary_group classify path "and" { b2 = B.and_ } ~nonzero_b:false
        o_and;
      int_binary_group classify path "or" { b2 = B.or_ } ~nonzero_b:false o_or;
      int_binary_group classify path "xor" { b2 = B.xor } ~nonzero_b:false o_xor;
    ]

  (* ───── Comparisons ───── *)

  let float_cmp_group classify path name (op : cmp) ref_fn =
    group name
      (List.map
         (fun (FDT d) ->
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, ta, tb, la, lb) ->
                   let expected = Array.map2 ref_fn la lb in
                   let got = op.c2 ta tb in
                   equal ~msg:lname (array bool) expected (F.to_array got))
                 (float_bin_cases d.dt d.pool d.pool)))
         float_dtypes)

  let int_cmp_group classify path name (op : cmp) refi =
    group name
      (List.map
         (fun (IDT d) ->
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, ta, tb, la, lb) ->
                   let expected =
                     Array.map2
                       (fun a b ->
                         refi ~bits:d.bits ~signed:d.signed (d.to_i64 a)
                           (d.to_i64 b))
                       la lb
                   in
                   equal ~msg:lname (array bool) expected
                     (F.to_array (op.c2 ta tb)))
                 (int_bin_cases d.dt d.pool d.pool)))
         int_dtypes)

  let ci_eq ~bits:_ ~signed:_ a b = a = b
  let ci_ne ~bits:_ ~signed:_ a b = a <> b

  let ci_lt ~bits ~signed a b =
    if signed then Int64.compare a b < 0
    else Int64.unsigned_compare (umag ~bits a) (umag ~bits b) < 0

  let ci_le ~bits ~signed a b =
    if signed then Int64.compare a b <= 0
    else Int64.unsigned_compare (umag ~bits a) (umag ~bits b) <= 0

  let comparison_tests classify path =
    [
      float_cmp_group classify path "cmpeq" { c2 = B.cmpeq } (fun a b -> a = b);
      float_cmp_group classify path "cmpne" { c2 = B.cmpne } (fun a b -> a <> b);
      float_cmp_group classify path "cmplt" { c2 = B.cmplt } (fun a b -> a < b);
      float_cmp_group classify path "cmple" { c2 = B.cmple } (fun a b -> a <= b);
      int_cmp_group classify path "cmpeq_i" { c2 = B.cmpeq } ci_eq;
      int_cmp_group classify path "cmpne_i" { c2 = B.cmpne } ci_ne;
      int_cmp_group classify path "cmplt_i" { c2 = B.cmplt } ci_lt;
      int_cmp_group classify path "cmple_i" { c2 = B.cmple } ci_le;
    ]

  (* ───── Reductions ───── *)

  let nanmax a b =
    if Float.is_nan a || Float.is_nan b then Float.nan
    else if a >= b then a
    else b

  let nanmin a b =
    if Float.is_nan a || Float.is_nan b then Float.nan
    else if a <= b then a
    else b

  let dedup l =
    List.fold_left
      (fun acc x -> if List.mem x acc then acc else acc @ [ x ])
      [] l

  let axes_options nd =
    if nd <= 1 then [ [| 0 |] ]
    else dedup [ [| 0 |]; [| nd - 1 |]; Array.init nd Fun.id ]

  let axes_count shape axes =
    Array.fold_left (fun acc ax -> acc * shape.(ax)) 1 axes

  let reduce_layouts (type a b) (dt : (a, b) Dtype.t) pool =
    List.filter
      (fun (n, _, _, _) -> n <> "scalar" && n <> "empty")
      (layouts dt pool)

  let reduce_generic (type x acc r) shape axes ~keepdims ~(init : acc)
      ~(f : acc -> x -> acc) ~(finish : acc -> r) (a : x array) :
      r array * int array =
    let nd = Array.length shape in
    let is_red = Array.make (max 1 nd) false in
    Array.iter (fun ax -> is_red.(ax) <- true) axes;
    let full = Array.mapi (fun i d -> if is_red.(i) then 1 else d) shape in
    let ostr = row_major_strides full in
    let onum = prod full in
    let acc = Array.make (max 1 onum) init in
    let n = prod shape in
    for lin = 0 to n - 1 do
      let idx = unravel lin shape in
      let oi = ref 0 in
      for i = 0 to nd - 1 do
        let c = if is_red.(i) then 0 else idx.(i) in
        oi := !oi + (c * ostr.(i))
      done;
      acc.(!oi) <- f acc.(!oi) a.(lin)
    done;
    let out = Array.init onum (fun i -> finish acc.(i)) in
    let oshape =
      if keepdims then full
      else
        Array.of_list
          (List.filteri (fun i _ -> not is_red.(i)) (Array.to_list shape))
    in
    (out, oshape)

  type red = { rd : 'a 'b. axes:int array -> ('a, 'b) B.t -> ('a, 'b) B.t }

  let float_reduction_group classify path name (op : red) ~init ~f ~scale =
    group name
      (List.map
         (fun (FDT d as fdt) ->
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   List.iter
                     (fun axes ->
                       let exp, oshape =
                         reduce_generic shape axes ~keepdims:false ~init ~f
                           ~finish:Fun.id logical
                       in
                       let k = float_of_int (max 1 (axes_count shape axes)) in
                       let tst =
                         if scale then
                           ftest ~rel:(d.a_rel *. k) ~abs:(d.a_abs *. k)
                         else f_arith fdt
                       in
                       let got = op.rd ~axes t in
                       let tag =
                         Printf.sprintf "%s ax=%s" lname
                           (String.concat ","
                              (List.map string_of_int (Array.to_list axes)))
                       in
                       equal ~msg:(tag ^ " shape") (array int) oshape
                         (F.shape got);
                       equal ~msg:tag (array tst) exp (F.to_array got))
                     (axes_options (Array.length shape)))
                 (reduce_layouts d.dt d.pool)))
         float_dtypes)

  let int_reduction_group classify path name (op : red) ~combine ~sum_mode =
    group name
      (List.map
         (fun (IDT d) ->
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               let f acc x =
                 match acc with
                 | None -> Some (d.to_i64 x)
                 | Some v ->
                     Some (combine ~bits:d.bits ~signed:d.signed v (d.to_i64 x))
               in
               let finish = function
                 | None -> d.of_i64 0L
                 | Some v ->
                     d.of_i64
                       (if sum_mode then wrap ~bits:d.bits ~signed:d.signed v
                        else v)
               in
               List.iter
                 (fun (lname, t, logical, shape) ->
                   List.iter
                     (fun axes ->
                       let exp, oshape =
                         reduce_generic shape axes ~keepdims:false ~init:None ~f
                           ~finish logical
                       in
                       let got = op.rd ~axes t in
                       let tag =
                         Printf.sprintf "%s ax=%s" lname
                           (String.concat ","
                              (List.map string_of_int (Array.to_list axes)))
                       in
                       equal ~msg:(tag ^ " shape") (array int) oshape
                         (F.shape got);
                       equal ~msg:tag (array d.tst) exp (F.to_array got))
                     (axes_options (Array.length shape)))
                 (reduce_layouts d.dt d.pool)))
         int_dtypes)

  let reduction_tests classify path =
    [
      float_reduction_group classify path "sum"
        { rd = (fun ~axes x -> B.reduce ~op:`Sum ~axes x) }
        ~init:0.0 ~f:( +. ) ~scale:true;
      float_reduction_group classify path "prod"
        { rd = (fun ~axes x -> B.reduce ~op:`Prod ~axes x) }
        ~init:1.0 ~f:( *. ) ~scale:true;
      float_reduction_group classify path "max"
        { rd = (fun ~axes x -> B.reduce ~op:`Max ~axes x) }
        ~init:Float.neg_infinity ~f:nanmax ~scale:false;
      float_reduction_group classify path "min"
        { rd = (fun ~axes x -> B.reduce ~op:`Min ~axes x) }
        ~init:Float.infinity ~f:nanmin ~scale:false;
      int_reduction_group classify path "sum"
        { rd = (fun ~axes x -> B.reduce ~op:`Sum ~axes x) }
        ~combine:o_add ~sum_mode:true;
      int_reduction_group classify path "max"
        { rd = (fun ~axes x -> B.reduce ~op:`Max ~axes x) }
        ~combine:o_max ~sum_mode:false;
      int_reduction_group classify path "min"
        { rd = (fun ~axes x -> B.reduce ~op:`Min ~axes x) }
        ~combine:o_min ~sum_mode:false;
    ]

  (* ───── argmax / argmin ─────

     NaN wins with the first index, and argmax/argmin agree on that rule; ties
     resolve to the first occurrence. *)

  let argreduce_float shape axis ~keepdims ~is_max (a : float array) :
      int32 array * int array =
    let nd = Array.length shape in
    let alen = shape.(axis) in
    let istr = row_major_strides shape in
    let full = Array.mapi (fun i d -> if i = axis then 1 else d) shape in
    let onum = prod full in
    let out = Array.make (max 1 onum) 0l in
    for oi = 0 to onum - 1 do
      let idx = unravel oi full in
      let base = ref 0 in
      for i = 0 to nd - 1 do
        base := !base + (idx.(i) * istr.(i))
      done;
      let at k = a.(!base + (k * istr.(axis))) in
      let best = ref 0 and nan_seen = ref (Float.is_nan (at 0)) in
      for k = 1 to alen - 1 do
        let v = at k in
        if Float.is_nan v then (
          if not !nan_seen then (
            best := k;
            nan_seen := true))
        else if not !nan_seen then
          if if is_max then v > at !best else v < at !best then best := k
      done;
      out.(oi) <- Int32.of_int !best
    done;
    let oshape =
      if keepdims then full
      else
        Array.of_list
          (List.filteri (fun i _ -> i <> axis) (Array.to_list shape))
    in
    (out, oshape)

  type arg = {
    ag :
      'a 'b.
      axis:int -> keepdims:bool -> ('a, 'b) B.t -> (int32, Dtype.int32_elt) B.t;
  }

  let argreduce_group classify path name (op : arg) ~is_max =
    group name
      (List.map
         (fun (FDT d) ->
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   let nd = Array.length shape in
                   List.iter
                     (fun axis ->
                       List.iter
                         (fun keepdims ->
                           let exp, oshape =
                             argreduce_float shape axis ~keepdims ~is_max
                               logical
                           in
                           let got = op.ag ~axis ~keepdims t in
                           let tag =
                             Printf.sprintf "%s ax=%d kd=%b" lname axis keepdims
                           in
                           equal ~msg:(tag ^ " shape") (array int) oshape
                             (F.shape got);
                           equal ~msg:tag (array int32) exp (F.to_array got))
                         [ false; true ])
                     (dedup [ 0; nd - 1 ]))
                 (reduce_layouts d.dt d.pool)))
         float_dtypes)

  let argreduce_tests classify path =
    [
      argreduce_group classify path "argmax" { ag = B.argmax } ~is_max:true;
      argreduce_group classify path "argmin" { ag = B.argmin } ~is_max:false;
    ]

  (* ───── Associative scan (cumulative) ───── *)

  let scan_ref (type x) shape axis ~(f : x -> x -> x) (a : x array) : x array =
    let istr = row_major_strides shape in
    let n = prod shape in
    let out = Array.copy a in
    for lin = 0 to n - 1 do
      let idx = unravel lin shape in
      if idx.(axis) > 0 then out.(lin) <- f out.(lin - istr.(axis)) a.(lin)
    done;
    out

  let scan_float_group classify path name op ~f =
    group name
      (List.map
         (fun (FDT d) ->
           let tst = ftest ~rel:(d.a_rel *. 6.0) ~abs:(d.a_abs *. 6.0) in
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   let nd = Array.length shape in
                   List.iter
                     (fun axis ->
                       let exp = scan_ref shape axis ~f logical in
                       let got = B.associative_scan ~axis ~op t in
                       let tag = Printf.sprintf "%s ax=%d" lname axis in
                       equal ~msg:(tag ^ " shape") (array int) shape
                         (F.shape got);
                       equal ~msg:tag (array tst) exp (F.to_array got))
                     (dedup [ 0; nd - 1 ]))
                 (reduce_layouts d.dt d.pool)))
         float_dtypes)

  let scan_int_group classify path name op ~combine ~wrap_mode =
    group name
      (List.map
         (fun (IDT d) ->
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   let nd = Array.length shape in
                   let a64 = Array.map d.to_i64 logical in
                   let f a b =
                     let r = combine ~bits:d.bits ~signed:d.signed a b in
                     if wrap_mode then wrap ~bits:d.bits ~signed:d.signed r
                     else r
                   in
                   List.iter
                     (fun axis ->
                       let exp =
                         Array.map d.of_i64 (scan_ref shape axis ~f a64)
                       in
                       let got = B.associative_scan ~axis ~op t in
                       let tag = Printf.sprintf "%s ax=%d" lname axis in
                       equal ~msg:tag (array d.tst) exp (F.to_array got))
                     (dedup [ 0; nd - 1 ]))
                 (reduce_layouts d.dt d.pool)))
         int_dtypes)

  let scan_tests classify path =
    [
      scan_float_group classify path "cumsum" `Sum ~f:( +. );
      scan_float_group classify path "cumprod" `Prod ~f:( *. );
      scan_float_group classify path "cummax" `Max ~f:nanmax;
      scan_float_group classify path "cummin" `Min ~f:nanmin;
      scan_int_group classify path "cumsum" `Sum ~combine:o_add ~wrap_mode:true;
      scan_int_group classify path "cummax" `Max ~combine:o_max ~wrap_mode:false;
      scan_int_group classify path "cummin" `Min ~combine:o_min ~wrap_mode:false;
    ]

  (* ───── Per-axis line helpers (sort, argsort) ───── *)

  let lines_of (type x) shape axis (a : x array) : x array list =
    let nd = Array.length shape in
    let istr = row_major_strides shape in
    let alen = shape.(axis) in
    let full = Array.mapi (fun i d -> if i = axis then 1 else d) shape in
    let onum = prod full in
    List.init onum (fun oi ->
        let idx = unravel oi full in
        let base = ref 0 in
        for i = 0 to nd - 1 do
          base := !base + (idx.(i) * istr.(i))
        done;
        Array.init alen (fun k -> a.(!base + (k * istr.(axis)))))

  let map_lines (type x) shape axis (transform : x array -> x array)
      (a : x array) : x array =
    let nd = Array.length shape in
    let istr = row_major_strides shape in
    let alen = shape.(axis) in
    let full = Array.mapi (fun i d -> if i = axis then 1 else d) shape in
    let onum = prod full in
    let out = Array.copy a in
    for oi = 0 to onum - 1 do
      let idx = unravel oi full in
      let base = ref 0 in
      for i = 0 to nd - 1 do
        base := !base + (idx.(i) * istr.(i))
      done;
      let line = Array.init alen (fun k -> a.(!base + (k * istr.(axis)))) in
      let line' = transform line in
      Array.iteri (fun k v -> out.(!base + (k * istr.(axis))) <- v) line'
    done;
    out

  (* NaN-last regardless of direction; ties keep any order (values compare
     equal). [cmp] orders non-NaN ascending. *)
  let sorted_line (type x) ~cmp ~is_nan ~descending (line : x array) : x array =
    let nans, rest = List.partition is_nan (Array.to_list line) in
    let s = List.sort cmp rest in
    let s = if descending then List.rev s else s in
    Array.of_list (s @ nans)

  (* ───── Sort ───── *)

  let sort_float_group classify path name ~descending =
    group name
      (List.map
         (fun (FDT d as fdt) ->
           let tst = f_arith fdt in
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   let nd = Array.length shape in
                   List.iter
                     (fun axis ->
                       let exp =
                         map_lines shape axis
                           (sorted_line ~cmp:Float.compare ~is_nan:Float.is_nan
                              ~descending)
                           logical
                       in
                       let got = B.sort ~axis ~descending t in
                       let tag =
                         Printf.sprintf "%s ax=%d desc=%b" lname axis descending
                       in
                       equal ~msg:(tag ^ " shape") (array int) shape
                         (F.shape got);
                       equal ~msg:tag (array tst) exp (F.to_array got))
                     (dedup [ 0; nd - 1 ]))
                 (reduce_layouts d.dt d.pool)))
         float_dtypes)

  let sort_int_group classify path name ~descending =
    group name
      (List.map
         (fun (IDT d) ->
           let cmp a b =
             if d.signed then Int64.compare a b
             else
               Int64.unsigned_compare (umag ~bits:d.bits a)
                 (umag ~bits:d.bits b)
           in
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   let nd = Array.length shape in
                   let a64 = Array.map d.to_i64 logical in
                   List.iter
                     (fun axis ->
                       let exp =
                         Array.map d.of_i64
                           (map_lines shape axis
                              (sorted_line ~cmp
                                 ~is_nan:(fun _ -> false)
                                 ~descending)
                              a64)
                       in
                       let got = B.sort ~axis ~descending t in
                       let tag =
                         Printf.sprintf "%s ax=%d desc=%b" lname axis descending
                       in
                       equal ~msg:tag (array d.tst) exp (F.to_array got))
                     (dedup [ 0; nd - 1 ]))
                 (reduce_layouts d.dt d.pool)))
         int_dtypes)

  (* argsort is checked by property: the returned indices are a permutation
     along the axis, and gathering by them reproduces the sorted values. This is
     robust to the tie-break rule (which the contract does not fix). *)
  let argsort_float_group classify path name ~descending =
    group name
      (List.map
         (fun (FDT d as fdt) ->
           let eq = Testable.equal (f_arith fdt) in
           case classify
             (path ^ "/" ^ name)
             d.name
             (fun () ->
               List.iter
                 (fun (lname, t, logical, shape) ->
                   let nd = Array.length shape in
                   List.iter
                     (fun axis ->
                       let idx = F.to_array (B.argsort ~axis ~descending t) in
                       let vlines = lines_of shape axis logical in
                       let ilines =
                         lines_of shape axis (Array.map Int32.to_int idx)
                       in
                       let tag =
                         Printf.sprintf "%s ax=%d desc=%b" lname axis descending
                       in
                       List.iter2
                         (fun vals iline ->
                           let alen = Array.length vals in
                           let perm = List.sort compare (Array.to_list iline) in
                           equal ~msg:(tag ^ " perm") (list int)
                             (List.init alen Fun.id) perm;
                           let reordered =
                             Array.map (fun j -> vals.(j)) iline
                           in
                           let expected =
                             sorted_line ~cmp:Float.compare ~is_nan:Float.is_nan
                               ~descending vals
                           in
                           Array.iteri
                             (fun k e ->
                               is_true ~msg:(tag ^ " reconstruct")
                                 (eq e reordered.(k)))
                             expected)
                         vlines ilines)
                     (dedup [ 0; nd - 1 ]))
                 (reduce_layouts d.dt d.pool)))
         float_dtypes)

  let sort_tests classify path =
    [
      sort_float_group classify path "sort_asc" ~descending:false;
      sort_float_group classify path "sort_desc" ~descending:true;
      sort_int_group classify path "sort_int_asc" ~descending:false;
      sort_int_group classify path "sort_int_desc" ~descending:true;
      argsort_float_group classify path "argsort_asc" ~descending:false;
      argsort_float_group classify path "argsort_desc" ~descending:true;
    ]

  (* ───── Ternary: where ───── *)

  let bpool =
    [|
      true;
      false;
      true;
      true;
      false;
      false;
      true;
      false;
      true;
      true;
      false;
      true;
      false;
      true;
      true;
      false;
      true;
      false;
    |]

  let where_tests classify path =
    let mkc shape data = F.create ctx F.bool shape data in
    let float_where =
      List.map
        (fun (FDT d as fdt) ->
          let tst = f_arith fdt in
          case classify (path ^ "/where") ("f_" ^ d.name) (fun () ->
              let a = F.create ctx d.dt [| 3; 4 |] (Array.sub d.pool 0 12) in
              let b = F.create ctx d.dt [| 3; 4 |] (Array.sub d.pool 4 12) in
              let sa = F.to_array a and sb = F.to_array b in
              let bc = Array.sub bpool 0 12 in
              let cond = mkc [| 3; 4 |] bc in
              (* contiguous condition *)
              let exp =
                Array.init 12 (fun i -> if bc.(i) then sa.(i) else sb.(i))
              in
              equal ~msg:"contig cond" (array tst) exp
                (F.to_array (B.where cond a b));
              (* strided (transposed) condition over transposed operands *)
              let cT = mkc [| 4; 3 |] (Array.sub bpool 0 12) in
              let condT = B.permute cT [| 1; 0 |] in
              let aT =
                B.permute
                  (F.create ctx d.dt [| 4; 3 |] (Array.sub d.pool 0 12))
                  [| 1; 0 |]
              in
              let bT =
                B.permute
                  (F.create ctx d.dt [| 4; 3 |] (Array.sub d.pool 4 12))
                  [| 1; 0 |]
              in
              let lc = permute_arr [| 4; 3 |] [| 1; 0 |] (F.to_array cT) in
              let la =
                permute_arr [| 4; 3 |] [| 1; 0 |]
                  (F.to_array
                     (F.create ctx d.dt [| 4; 3 |] (Array.sub d.pool 0 12)))
              in
              let lb =
                permute_arr [| 4; 3 |] [| 1; 0 |]
                  (F.to_array
                     (F.create ctx d.dt [| 4; 3 |] (Array.sub d.pool 4 12)))
              in
              let expT =
                Array.init 12 (fun i -> if lc.(i) then la.(i) else lb.(i))
              in
              equal ~msg:"strided cond" (array tst) expT
                (F.to_array (B.where condT aT bT))))
        float_dtypes
    in
    [ group "where" float_where ]

  (* ───── Movement (backend-level view guarantees) ───── *)

  let movement_tests classify path =
    let f = F.float64 in
    let mk shape data = F.create ctx f shape data in
    let d12 = Array.init 12 (fun i -> float_of_int (i + 1)) in
    [
      group "movement"
        [
          case classify path "contiguous-materializes" (fun () ->
              let strided = B.permute (mk [| 3; 4 |] d12) [| 1; 0 |] in
              let c = B.contiguous strided in
              is_true ~msg:"is c-contiguous" (View.is_c_contiguous (B.view c));
              equal ~msg:"offset 0" int 0 (View.offset (B.view c));
              equal ~msg:"values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                (permute_arr [| 3; 4 |] [| 1; 0 |] d12)
                (F.to_array c));
          case classify path "full-is-contiguous" (fun () ->
              let t = B.full ctx f [| 2; 3 |] 7.0 in
              is_true ~msg:"contiguous" (View.is_c_contiguous (B.view t));
              equal ~msg:"shape" (array int) [| 2; 3 |] (F.shape t);
              equal ~msg:"values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                (Array.make 6 7.0) (F.to_array t));
          case classify path "copy-preserves-values" (fun () ->
              let strided = B.flip (mk [| 3; 4 |] d12) [| false; true |] in
              let c = B.copy strided in
              equal ~msg:"values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                (reverse_last 3 4 d12) (F.to_array c));
          case classify path "pad-constant" (fun () ->
              let t = mk [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
              let p = B.pad t [| (1, 0); (0, 2) |] (-1.0) in
              equal ~msg:"shape" (array int) [| 3; 4 |] (F.shape p);
              equal ~msg:"values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                [| -1.; -1.; -1.; -1.; 1.; 2.; -1.; -1.; 3.; 4.; -1.; -1. |]
                (F.to_array p));
          case classify path "cat-axis0" (fun () ->
              let a = mk [| 1; 2 |] [| 1.; 2. |] in
              let b = mk [| 2; 2 |] [| 3.; 4.; 5.; 6. |] in
              let c = B.cat [ a; b ] ~axis:0 in
              equal ~msg:"shape" (array int) [| 3; 2 |] (F.shape c);
              equal ~msg:"values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                [| 1.; 2.; 3.; 4.; 5.; 6. |]
                (F.to_array c));
          case classify path "cat-axis1-strided" (fun () ->
              let a =
                B.permute (mk [| 2; 2 |] [| 1.; 3.; 2.; 4. |]) [| 1; 0 |]
              in
              let b = mk [| 2; 1 |] [| 5.; 6. |] in
              let c = B.cat [ a; b ] ~axis:1 in
              equal ~msg:"shape" (array int) [| 2; 3 |] (F.shape c);
              equal ~msg:"values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                [| 1.; 2.; 5.; 3.; 4.; 6. |]
                (F.to_array c));
          case classify path "assign-strided-src" (fun () ->
              (* dst contiguous, src a transposed view; assign must respect both
                 layouts and write src's logical content into dst. *)
              let dst = mk [| 3; 4 |] (Array.make 12 0.0) in
              let src = B.permute (mk [| 4; 3 |] d12) [| 1; 0 |] in
              B.assign dst src;
              equal ~msg:"values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                (permute_arr [| 4; 3 |] [| 1; 0 |] d12)
                (F.to_array dst));
          case classify path "assign-strided-dst" (fun () ->
              (* dst is a transposed view over a fresh buffer; assign must
                 scatter src through dst's strides. Read back through the same
                 buffer (contiguized) to confirm the underlying storage was
                 written. *)
              let base = mk [| 4; 3 |] (Array.make 12 0.0) in
              let dst = B.permute base [| 1; 0 |] in
              let src = mk [| 3; 4 |] d12 in
              B.assign dst src;
              equal ~msg:"logical values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                d12 (F.to_array dst);
              (* base holds the transpose of d12 in its own row-major order *)
              equal ~msg:"buffer layout"
                (array (ftest ~rel:0.0 ~abs:0.0))
                (permute_arr [| 3; 4 |] [| 1; 0 |] d12)
                (F.to_array base));
          case classify path "assign-int4-preserves-tail-nibble" (fun () ->
              (* A packed prefix ends halfway through the destination's last
                 byte. Assign must update the low nibble without clobbering the
                 live element stored in the high nibble. *)
              let base = F.create ctx F.int4 [| 4 |] [| 1; 2; 3; 4 |] in
              let prefix = B.shrink base [| (0, 3) |] in
              let src = F.create ctx F.int4 [| 3 |] [| 5; 6; 7 |] in
              B.assign prefix src;
              equal ~msg:"packed neighbor preserved" (array int)
                [| 5; 6; 7; 4 |] (F.to_array base));
          case classify path "reshape-split-merge" (fun () ->
              (* B.reshape is view-only: it splits/merges dims of a contiguous
                 tensor with no copy and preserves row-major order. (Reshaping a
                 non-composable strided view raises "call contiguous() first" —
                 the frontend, not the backend, owns that copy.) *)
              let t = mk [| 3; 4 |] d12 in
              let split = B.reshape t [| 3; 2; 2 |] in
              equal ~msg:"split shape" (array int) [| 3; 2; 2 |] (F.shape split);
              equal ~msg:"split values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                d12 (F.to_array split);
              let merged = B.reshape split [| 12 |] in
              equal ~msg:"merge shape" (array int) [| 12 |] (F.shape merged);
              equal ~msg:"merge values"
                (array (ftest ~rel:0.0 ~abs:0.0))
                d12 (F.to_array merged));
        ];
    ]

  (* ───── Cast ───── *)

  let cast_tests classify path =
    let exact = array (ftest ~rel:0.0 ~abs:0.0) in
    [
      group "cast"
        [
          case classify path "f64->i32-trunc" (fun () ->
              let t =
                F.create ctx F.float64 [| 5 |] [| 1.9; -1.9; 2.5; -2.5; 0.0 |]
              in
              equal ~msg:"trunc toward zero" (array int32)
                [| 1l; -1l; 2l; -2l; 0l |]
                (F.to_array (B.cast ~dtype:F.int32 t)));
          case classify path "i32->f64" (fun () ->
              let t = F.create ctx F.int32 [| 3 |] [| 1l; -2l; 300l |] in
              equal ~msg:"widen" exact [| 1.; -2.; 300. |]
                (F.to_array (B.cast ~dtype:F.float64 t)));
          case classify path "f64->f32-round" (fun () ->
              let t =
                F.create ctx F.float64 [| 2 |] [| 1.0000001; 3.14159265358979 |]
              in
              let got = F.to_array (B.cast ~dtype:F.float32 t) in
              equal ~msg:"f32 precision"
                (array (ftest ~rel:1e-6 ~abs:1e-6))
                [| 1.0000001; 3.14159265358979 |]
                got);
          case classify path "f64->u8-saturate" (fun () ->
              let t =
                F.create ctx F.float64 [| 4 |] [| 300.0; -5.0; 128.0; 255.0 |]
              in
              equal ~msg:"clamp to [0,255]" (array int) [| 255; 0; 128; 255 |]
                (F.to_array (B.cast ~dtype:F.uint8 t)));
          case classify path "f64->i32-saturate-inf" (fun () ->
              let t =
                F.create ctx F.float64 [| 2 |]
                  [| Float.infinity; Float.neg_infinity |]
              in
              equal ~msg:"saturate to i32 range" (array int32)
                [| Int32.max_int; Int32.min_int |]
                (F.to_array (B.cast ~dtype:F.int32 t)));
          case classify path "f64->i32-nan-zero" (fun () ->
              (* Float-to-int saturates with no UB; NaN maps to 0. *)
              let t =
                F.create ctx F.float64 [| 3 |] [| Float.nan; 1.0; -1.0 |]
              in
              equal ~msg:"nan -> 0" (array int32) [| 0l; 1l; -1l |]
                (F.to_array (B.cast ~dtype:F.int32 t)));
          case classify path "f64->u8-nan-zero" (fun () ->
              let t =
                F.create ctx F.float64 [| 3 |] [| Float.nan; 10.0; 20.0 |]
              in
              equal ~msg:"nan -> 0" (array int) [| 0; 10; 20 |]
                (F.to_array (B.cast ~dtype:F.uint8 t)));
          case classify path "f64->i8-saturate" (fun () ->
              (* clamp to the destination dtype's range, not i32's *)
              let t =
                F.create ctx F.float64 [| 4 |]
                  [| 1000.0; -1000.0; 127.0; -128.0 |]
              in
              equal ~msg:"clamp to [-128,127]" (array int)
                [| 127; -128; 127; -128 |]
                (F.to_array (B.cast ~dtype:F.int8 t)));
          case classify path "f32->f64-widen-exact" (fun () ->
              let t = F.create ctx F.float32 [| 3 |] [| 0.5; -2.25; 3.75 |] in
              equal ~msg:"exact widen" exact [| 0.5; -2.25; 3.75 |]
                (F.to_array (B.cast ~dtype:F.float64 t)));
          case classify path "f64->complex128" (fun () ->
              (* real -> complex sets the imaginary part to 0; also underwrites
                 the eig residual oracle, which casts a real matrix to
                 complex. *)
              let z re im = { Complex.re; im } in
              let t = F.create ctx F.float64 [| 3 |] [| 1.5; -2.0; 3.0 |] in
              equal ~msg:"im=0"
                (array (ctest ~rel:0.0 ~abs:0.0))
                [| z 1.5 0.0; z (-2.0) 0.0; z 3.0 0.0 |]
                (F.to_array (B.cast ~dtype:F.complex128 t)));
          case classify path "int8<->int4-roundtrip" (fun () ->
              (* Int4 is storage-only, supported in cast (pack/unpack). In-range
                 values round-trip exactly. *)
              let t = F.create ctx F.int8 [| 5 |] [| -8; 7; 0; 3; -4 |] in
              equal ~msg:"in-range roundtrip" (array int) [| -8; 7; 0; 3; -4 |]
                (F.to_array (B.cast ~dtype:F.int8 (B.cast ~dtype:F.int4 t))));
          case classify path "uint8<->uint4-roundtrip" (fun () ->
              let t = F.create ctx F.uint8 [| 5 |] [| 0; 15; 7; 3; 8 |] in
              equal ~msg:"in-range roundtrip" (array int) [| 0; 15; 7; 3; 8 |]
                (F.to_array (B.cast ~dtype:F.uint8 (B.cast ~dtype:F.uint4 t))));
        ];
    ]

  (* ───── Threefry ─────

     Independent OCaml port of threefry2x32-20 (Int32 arithmetic wraps mod 2^32,
     matching uint32). The frontend consumes int32 key/counter tensors as
     consecutive 2-word vectors. Anchored to the Random123 known-answer
     vectors. *)

  let rotl32 x r =
    Int32.logor (Int32.shift_left x r) (Int32.shift_right_logical x (32 - r))

  let threefry2x32 k0 k1 c0 c1 =
    let ks2 = Int32.logxor (Int32.logxor 0x1BD11BDAl k0) k1 in
    let ks = [| k0; k1; ks2 |] in
    let x0 = ref (Int32.add c0 k0) and x1 = ref (Int32.add c1 k1) in
    let rots = [| 13; 15; 26; 6; 17; 29; 16; 24 |] in
    for r = 0 to 19 do
      x0 := Int32.add !x0 !x1;
      x1 := rotl32 !x1 rots.(r mod 8);
      x1 := Int32.logxor !x1 !x0;
      if (r + 1) mod 4 = 0 then begin
        let s = (r + 1) / 4 in
        x0 := Int32.add !x0 ks.(s mod 3);
        x1 := Int32.add (Int32.add !x1 ks.((s + 1) mod 3)) (Int32.of_int s)
      end
    done;
    (!x0, !x1)

  let threefry_ref (key : int32 array) (ctr : int32 array) : int32 array =
    let out = Array.copy key in
    let v = Array.length key / 2 in
    for i = 0 to v - 1 do
      let r0, r1 =
        threefry2x32 key.(2 * i) key.((2 * i) + 1) ctr.(2 * i) ctr.((2 * i) + 1)
      in
      out.(2 * i) <- r0;
      out.((2 * i) + 1) <- r1
    done;
    out

  let threefry_tests classify path =
    let mk data = F.create ctx F.int32 [| Array.length data |] data in
    let kat k0 k1 c0 c1 e0 e1 name =
      case classify path name (fun () ->
          let got =
            F.to_array (B.threefry (mk [| k0; k1 |]) (mk [| c0; c1 |]))
          in
          equal ~msg:"kat" (array int32) [| e0; e1 |] got)
    in
    let prng =
      (* deterministic filler, avoids a real RNG dependency *)
      let s = ref 0x243F6A88l in
      fun () ->
        s := Int32.add (Int32.mul !s 1103515245l) 12345l;
        !s
    in
    [
      group "threefry"
        [
          kat 0l 0l 0l 0l 1797259609l (-1715843330l) "kat-zero";
          kat (-1l) (-1l) (-1l) (-1l) 481924860l (-1157616665l) "kat-ones";
          kat 0x13198a2el 0x03707344l 0x243f6a88l 0x85a308d3l (-997049700l)
            1212020640l "kat-pi";
          case classify path "reference-agreement" (fun () ->
              (* the backend requires the 2-word vector to be the last axis *)
              let n = 16 in
              let key = Array.init n (fun _ -> prng ()) in
              let ctr = Array.init n (fun _ -> prng ()) in
              let exp = threefry_ref key ctr in
              let kt = F.create ctx F.int32 [| 8; 2 |] key in
              let ct = F.create ctx F.int32 [| 8; 2 |] ctr in
              let got = F.to_array (B.threefry kt ct) in
              equal ~msg:"vs reference" (array int32) exp got);
          case classify path "reference-agreement-2d" (fun () ->
              let key = Array.init 12 (fun _ -> prng ()) in
              let ctr = Array.init 12 (fun _ -> prng ()) in
              let exp = threefry_ref key ctr in
              let kt = F.create ctx F.int32 [| 6; 2 |] key in
              let ct = F.create ctx F.int32 [| 6; 2 |] ctr in
              let got = F.to_array (B.threefry kt ct) in
              equal ~msg:"vs reference" (array int32) exp got);
        ];
    ]

  (* ───── Gather / Scatter ───── *)

  let gather_tests classify path =
    let data =
      F.create ctx F.float64 [| 3; 4 |]
        (Array.init 12 (fun i -> float_of_int i))
    in
    let sdata = F.to_array data in
    let ftst = ftest ~rel:0.0 ~abs:0.0 in
    [
      group "gather"
        [
          case classify (path ^ "/gather") "axis1" (fun () ->
              (* out[r,c] = data[r, idx[r,c]] *)
              let idx_arr = [| 3l; 0l; 1l; 3l; 0l; 2l |] in
              let idx = F.create ctx F.int32 [| 3; 2 |] idx_arr in
              let got = B.gather data idx ~axis:1 in
              let exp =
                Array.init 6 (fun i ->
                    let r = i / 2 in
                    sdata.((r * 4) + Int32.to_int idx_arr.(i)))
              in
              equal ~msg:"shape" (array int) [| 3; 2 |] (F.shape got);
              equal ~msg:"values" (array ftst) exp (F.to_array got));
          case classify (path ^ "/gather") "axis0" (fun () ->
              (* out[r,c] = data[idx[r,c], c] *)
              let idx_arr = [| 2l; 0l; 1l; 2l; 0l; 1l; 2l; 0l |] in
              let idx = F.create ctx F.int32 [| 2; 4 |] idx_arr in
              let got = B.gather data idx ~axis:0 in
              let exp =
                Array.init 8 (fun i ->
                    let c = i mod 4 in
                    sdata.((Int32.to_int idx_arr.(i) * 4) + c))
              in
              equal ~msg:"shape" (array int) [| 2; 4 |] (F.shape got);
              equal ~msg:"values" (array ftst) exp (F.to_array got));
        ];
    ]

  let scatter_tests classify path =
    let ftst = ftest ~rel:0.0 ~abs:0.0 in
    let template = F.create ctx F.float64 [| 3; 4 |] (Array.make 12 0.0) in
    [
      group "scatter"
        [
          case classify (path ^ "/scatter") "set-axis1" (fun () ->
              let idx =
                F.create ctx F.int32 [| 3; 2 |] [| 0l; 2l; 1l; 3l; 0l; 1l |]
              in
              let updates =
                F.create ctx F.float64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
              in
              let got =
                B.scatter ~mode:`Set ~unique_indices:false template ~indices:idx
                  ~updates ~axis:1
              in
              let exp = [| 1.; 0.; 2.; 0.; 0.; 3.; 0.; 4.; 5.; 6.; 0.; 0. |] in
              equal ~msg:"shape" (array int) [| 3; 4 |] (F.shape got);
              equal ~msg:"values" (array ftst) exp (F.to_array got));
          case classify (path ^ "/scatter") "add-duplicate" (fun () ->
              (* duplicate target (0,1) accumulates 10 + 40 *)
              let idx = F.create ctx F.int32 [| 1; 4 |] [| 1l; 1l; 2l; 1l |] in
              let updates =
                F.create ctx F.float64 [| 1; 4 |] [| 10.; 40.; 20.; 5. |]
              in
              let tmpl = F.create ctx F.float64 [| 1; 4 |] (Array.make 4 0.0) in
              let got =
                B.scatter ~mode:`Add ~unique_indices:false tmpl ~indices:idx
                  ~updates ~axis:1
              in
              equal ~msg:"accumulate" (array ftst) [| 0.; 55.; 20.; 0. |]
                (F.to_array got));
          case classify (path ^ "/scatter") "set-duplicate-last-wins" (fun () ->
              (* two updates target column 1; `Set keeps the last in scan order,
                 and untouched template cells are preserved *)
              let idx = F.create ctx F.int32 [| 1; 3 |] [| 1l; 1l; 3l |] in
              let updates =
                F.create ctx F.float64 [| 1; 3 |] [| 10.; 40.; 20. |]
              in
              let tmpl = F.create ctx F.float64 [| 1; 4 |] (Array.make 4 7.0) in
              let got =
                B.scatter ~mode:`Set ~unique_indices:false tmpl ~indices:idx
                  ~updates ~axis:1
              in
              equal ~msg:"last wins" (array ftst) [| 7.; 40.; 7.; 20. |]
                (F.to_array got));
          case classify (path ^ "/scatter") "set-unique-indices" (fun () ->
              (* unique_indices:true (a hint the indices are distinct); each
                 update lands at its column, untouched cells keep the template
                 value. *)
              let idx = F.create ctx F.int32 [| 1; 3 |] [| 0l; 2l; 3l |] in
              let updates =
                F.create ctx F.float64 [| 1; 3 |] [| 10.; 20.; 30. |]
              in
              let tmpl = F.create ctx F.float64 [| 1; 4 |] (Array.make 4 5.0) in
              let got =
                B.scatter ~mode:`Set ~unique_indices:true tmpl ~indices:idx
                  ~updates ~axis:1
              in
              equal ~msg:"unique set" (array ftst) [| 10.; 5.; 20.; 30. |]
                (F.to_array got));
        ];
    ]

  (* ───── Window: unfold / fold ───── *)

  let unfold_tests classify path =
    let ftst = ftest ~rel:0.0 ~abs:0.0 in
    [
      group "window"
        [
          case classify path "unfold-1d" (fun () ->
              let x =
                F.create ctx F.float64 [| 2; 5 |]
                  (Array.init 10 (fun i -> float_of_int i))
              in
              let sx = F.to_array x in
              let got =
                B.unfold x ~kernel_size:[| 2 |] ~stride:[| 1 |]
                  ~dilation:[| 1 |]
                  ~padding:[| (0, 0) |]
              in
              (* output [2; prod_kernel=2; L=4]; out[n,ki,w] = x[n, w+ki] *)
              equal ~msg:"shape" (array int) [| 2; 2; 4 |] (F.shape got);
              let exp =
                Array.init
                  (2 * 2 * 4)
                  (fun lin ->
                    let n = lin / 8 and rem = lin mod 8 in
                    let ki = rem / 4 and w = rem mod 4 in
                    sx.((n * 5) + w + ki))
              in
              equal ~msg:"values" (array ftst) exp (F.to_array got));
          case classify path "unfold-1d-dilation-padding" (fun () ->
              (* kernel 2, stride 1, dilation 2, padding (1,1): two taps spaced
                 by the dilation, zero-padded outside the input. *)
              let x =
                F.create ctx F.float64 [| 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |]
              in
              let got =
                B.unfold x ~kernel_size:[| 2 |] ~stride:[| 1 |]
                  ~dilation:[| 2 |]
                  ~padding:[| (1, 1) |]
              in
              let padded = [| 0.; 1.; 2.; 3.; 4.; 5.; 0. |] in
              let l = 5 in
              let exp =
                Array.init (2 * l) (fun lin ->
                    let ki = lin / l and w = lin mod l in
                    padded.(w + (ki * 2)))
              in
              equal ~msg:"shape" (array int) [| 1; 2; l |] (F.shape got);
              equal ~msg:"values" (array ftst) exp (F.to_array got));
          case classify path "fold-inverts-unfold-nonoverlap" (fun () ->
              (* stride = kernel: each input element lands in exactly one
                 window, so fold (sum of overlaps) reconstructs the input
                 exactly. *)
              let x =
                F.create ctx F.float64 [| 2; 6 |]
                  (Array.init 12 (fun i -> float_of_int (i + 1)))
              in
              let u =
                B.unfold x ~kernel_size:[| 2 |] ~stride:[| 2 |]
                  ~dilation:[| 1 |]
                  ~padding:[| (0, 0) |]
              in
              let f =
                B.fold u ~output_size:[| 6 |] ~kernel_size:[| 2 |]
                  ~stride:[| 2 |] ~dilation:[| 1 |]
                  ~padding:[| (0, 0) |]
              in
              equal ~msg:"shape" (array int) [| 2; 6 |] (F.shape f);
              equal ~msg:"reconstruct" (array ftst) (F.to_array x)
                (F.to_array f));
          (let overlap_fold_check leading len k s =
             (* fold(unfold(x))[i] = coverage[i] * x[i], where coverage[i] is
                how many length-k windows (stride s) cover i. This is the col2im
                accumulation path (AppxA#1 race site). *)
             let x =
               F.create ctx F.float64 [| leading; len |]
                 (Array.init (leading * len) (fun i ->
                      float_of_int ((i mod 13) + 1)))
             in
             let l = ((len - k) / s) + 1 in
             let u =
               B.unfold x ~kernel_size:[| k |] ~stride:[| s |] ~dilation:[| 1 |]
                 ~padding:[| (0, 0) |]
             in
             let f =
               B.fold u ~output_size:[| len |] ~kernel_size:[| k |]
                 ~stride:[| s |] ~dilation:[| 1 |]
                 ~padding:[| (0, 0) |]
             in
             let coverage j =
               let c = ref 0 in
               for w = 0 to l - 1 do
                 if j >= w * s && j < (w * s) + k then incr c
               done;
               !c
             in
             let sx = F.to_array x in
             let exp =
               Array.init (leading * len) (fun lin ->
                   sx.(lin) *. float_of_int (coverage (lin mod len)))
             in
             equal ~msg:"shape" (array int) [| leading; len |] (F.shape f);
             equal ~msg:"sum of overlaps" (array ftst) exp (F.to_array f)
           in
           group "fold-overlap"
             [
               case classify (path ^ "/fold-overlap") "small-value-gate"
                 (fun () ->
                   (* small: serial, deterministic — a pure value gate *)
                   overlap_fold_check 4 8 3 1);
               case classify (path ^ "/fold-overlap") "batch-parallel"
                 (fun () ->
                   (* A scaled overlap-fold gate with many independent leading
                      rows, suitable for a backend's parallel path. *)
                   overlap_fold_check 64 32 3 1);
               case classify (path ^ "/fold-overlap") "race-witness" (fun () ->
                   (* leading = 1, large L, kernel > stride: adjacent windows
                      overlap within one row. This catches any parallel
                      implementation that partitions windows without preserving
                      exclusive output ownership. *)
                   overlap_fold_check 1 2048 4 1);
             ]);
        ];
    ]

  (* ───── Matmul ───── *)

  let matmul_ref2 m k n (a : float array) (b : float array) : float array =
    let c = Array.make (m * n) 0.0 in
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let acc = ref 0.0 in
        for p = 0 to k - 1 do
          acc := !acc +. (a.((i * k) + p) *. b.((p * n) + j))
        done;
        c.((i * n) + j) <- !acc
      done
    done;
    c

  let matmul_tests classify path =
    let mkf shape data = F.create ctx F.float64 shape data in
    let seq n = Array.init n (fun i -> float_of_int ((i mod 7) - 3)) in
    let tol k = ftest ~rel:(1e-11 *. float_of_int (k + 1)) ~abs:1e-11 in
    [
      group "matmul"
        [
          case classify path "2d" (fun () ->
              let m = 3 and k = 5 and n = 4 in
              let a = mkf [| m; k |] (seq (m * k)) in
              let b = mkf [| k; n |] (seq (k * n)) in
              let exp = matmul_ref2 m k n (F.to_array a) (F.to_array b) in
              let got = B.matmul a b in
              equal ~msg:"shape" (array int) [| m; n |] (F.shape got);
              equal ~msg:"values" (array (tol k)) exp (F.to_array got));
          case classify path "2d-transposed-lhs" (fun () ->
              (* A given as a transposed (non-contiguous) view *)
              let m = 3 and k = 4 and n = 2 in
              let a_base = mkf [| k; m |] (seq (k * m)) in
              let a = B.permute a_base [| 1; 0 |] in
              let la = permute_arr [| k; m |] [| 1; 0 |] (F.to_array a_base) in
              let b = mkf [| k; n |] (seq (k * n)) in
              let exp = matmul_ref2 m k n la (F.to_array b) in
              let got = B.matmul a b in
              equal ~msg:"shape" (array int) [| m; n |] (F.shape got);
              equal ~msg:"values" (array (tol k)) exp (F.to_array got));
          case classify path "2d-both-transposed" (fun () ->
              (* both operands transposed views — neither has a unit-stride
                 row *)
              let m = 3 and k = 4 and n = 2 in
              let a_base = mkf [| k; m |] (seq (k * m)) in
              let b_base = mkf [| n; k |] (seq (n * k)) in
              let a = B.permute a_base [| 1; 0 |] in
              let b = B.permute b_base [| 1; 0 |] in
              let la = permute_arr [| k; m |] [| 1; 0 |] (F.to_array a_base) in
              let lb = permute_arr [| n; k |] [| 1; 0 |] (F.to_array b_base) in
              let exp = matmul_ref2 m k n la lb in
              let got = B.matmul a b in
              equal ~msg:"shape" (array int) [| m; n |] (F.shape got);
              equal ~msg:"values" (array (tol k)) exp (F.to_array got));
          case classify path "batched" (fun () ->
              let bsz = 2 and m = 2 and k = 3 and n = 2 in
              let a = mkf [| bsz; m; k |] (seq (bsz * m * k)) in
              let b = mkf [| bsz; k; n |] (seq (bsz * k * n)) in
              let sa = F.to_array a and sb = F.to_array b in
              let exp =
                Array.concat
                  (List.init bsz (fun bi ->
                       matmul_ref2 m k n
                         (Array.sub sa (bi * m * k) (m * k))
                         (Array.sub sb (bi * k * n) (k * n))))
              in
              let got = B.matmul a b in
              equal ~msg:"shape" (array int) [| bsz; m; n |] (F.shape got);
              equal ~msg:"values" (array (tol k)) exp (F.to_array got));
          case classify path "transposed-rhs" (fun () ->
              let m = 3 and k = 4 and n = 2 in
              let a = mkf [| m; k |] (seq (m * k)) in
              let b_base = mkf [| n; k |] (seq (n * k)) in
              let b = B.permute b_base [| 1; 0 |] in
              let lb = permute_arr [| n; k |] [| 1; 0 |] (F.to_array b_base) in
              let exp = matmul_ref2 m k n (F.to_array a) lb in
              let got = B.matmul a b in
              equal ~msg:"shape" (array int) [| m; n |] (F.shape got);
              equal ~msg:"values" (array (tol k)) exp (F.to_array got));
          case classify path "batch-broadcast" (fun () ->
              (* a is batched, b is 2-D and broadcast across the batch (stride-0
                 batch dim on b) *)
              let bsz = 2 and m = 2 and k = 3 and n = 2 in
              let a = mkf [| bsz; m; k |] (seq (bsz * m * k)) in
              let b = mkf [| k; n |] (seq (k * n)) in
              let sa = F.to_array a and sb = F.to_array b in
              let exp =
                Array.concat
                  (List.init bsz (fun bi ->
                       matmul_ref2 m k n (Array.sub sa (bi * m * k) (m * k)) sb))
              in
              let got = B.matmul a b in
              equal ~msg:"shape" (array int) [| bsz; m; n |] (F.shape got);
              equal ~msg:"values" (array (tol k)) exp (F.to_array got));
          case classify path "f32" (fun () ->
              let m = 3 and k = 5 and n = 4 in
              let a = F.create ctx F.float32 [| m; k |] (seq (m * k)) in
              let b = F.create ctx F.float32 [| k; n |] (seq (k * n)) in
              let exp = matmul_ref2 m k n (F.to_array a) (F.to_array b) in
              let got = B.matmul a b in
              equal ~msg:"shape" (array int) [| m; n |] (F.shape got);
              equal ~msg:"values"
                (array (ftest ~rel:1e-5 ~abs:1e-5))
                exp (F.to_array got));
        ];
    ]

  (* ───── Linear algebra (residual properties) ─────

     Factorizations are checked by residual bounds and orthogonality rather than
     against a stored oracle: correctness is a property of the reconstruction.
     The backend's own matmul (independently gated above) closes the loop. *)

  let resid (a : float array) (b : float array) =
    let m = ref 0.0 in
    Array.iteri (fun i x -> m := Float.max !m (Float.abs (x -. b.(i)))) a;
    !m

  (* Relative residual ‖ref - x‖ / ‖ref‖ (Frobenius / max-abs), so the gate is
     scale-independent and holds if the fixtures change. *)
  let fnorm a = sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0.0 a)
  let rel_resid ref_ x = resid ref_ x /. Float.max 1e-30 (fnorm ref_)

  let identity n =
    Array.init (n * n) (fun i -> if i / n = i mod n then 1.0 else 0.0)

  let diag n (v : float array) =
    Array.init (n * n) (fun i -> if i / n = i mod n then v.(i / n) else 0.0)

  let transpose_arr r c a = permute_arr [| r; c |] [| 1; 0 |] a

  let cresid (a : Complex.t array) (b : Complex.t array) =
    let m = ref 0.0 in
    Array.iteri
      (fun i x ->
        m :=
          Float.max !m
            (Float.max
               (Float.abs (x.Complex.re -. b.(i).Complex.re))
               (Float.abs (x.Complex.im -. b.(i).Complex.im))))
      a;
    !m

  let cfnorm a =
    sqrt
      (Array.fold_left
         (fun s w ->
           s +. (w.Complex.re *. w.Complex.re) +. (w.Complex.im *. w.Complex.im))
         0.0 a)

  let crel_resid ref_ x = cresid ref_ x /. Float.max 1e-30 (cfnorm ref_)
  let mk64 shape data = F.create ctx F.float64 shape data
  let mm a b = B.matmul a b
  let tr a = B.permute a [| 1; 0 |]

  let linalg_tests classify path =
    let tol = 1e-9 in
    (* relative residual gate: ‖A - reconstruction‖ / ‖A‖ *)
    let rok ref_ recon = rel_resid ref_ recon <= tol in
    (* An arbitrary full-rank 4x3 and a related SPD 4x4. *)
    let m4x3 = [| 1.; 2.; -1.; 0.; 3.; 1.; 2.; -2.; 1.; 1.; 0.; 4. |] in
    let mk_spd n gen =
      let mm_ = matmul_ref2 n n n in
      let mt = transpose_arr n n gen in
      let g = mm_ gen mt in
      Array.mapi
        (fun i x -> if i / n = i mod n then x +. float_of_int n else x)
        g
    in
    let sq4 =
      [| 1.; 2.; 0.; 1.; 0.; 1.; 1.; 0.; 2.; 0.; 1.; 1.; 1.; 1.; 0.; 3. |]
    in
    let spd4 = mk_spd 4 sq4 in
    [
      group "linalg"
        [
          case classify path "cholesky-lower" (fun () ->
              let a = mk64 [| 4; 4 |] spd4 in
              let l = B.cholesky ~upper:false a in
              let la = F.to_array l in
              (* lower triangular *)
              Array.iteri
                (fun i v ->
                  if i mod 4 > i / 4 then
                    is_true ~msg:"lower zero" (Float.abs v <= tol))
                la;
              let recon = F.to_array (mm l (tr l)) in
              is_true
                ~msg:(Printf.sprintf "LLt=A rel %g" (rel_resid spd4 recon))
                (rok spd4 recon));
          case classify path "cholesky-upper" (fun () ->
              let a = mk64 [| 4; 4 |] spd4 in
              let u = B.cholesky ~upper:true a in
              let recon = F.to_array (mm (tr u) u) in
              is_true
                ~msg:(Printf.sprintf "UtU=A rel %g" (rel_resid spd4 recon))
                (rok spd4 recon));
          case classify path "qr-reduced" (fun () ->
              let a = mk64 [| 4; 3 |] m4x3 in
              let q, r = B.qr ~reduced:true a in
              equal ~msg:"Q shape" (array int) [| 4; 3 |] (F.shape q);
              equal ~msg:"R shape" (array int) [| 3; 3 |] (F.shape r);
              let recon = F.to_array (mm q r) in
              is_true
                ~msg:(Printf.sprintf "QR=A rel %g" (rel_resid m4x3 recon))
                (rok m4x3 recon);
              let qtq = F.to_array (mm (tr q) q) in
              is_true ~msg:"QtQ=I" (rok (identity 3) qtq);
              let ra = F.to_array r in
              Array.iteri
                (fun i v ->
                  if i / 3 > i mod 3 then
                    is_true ~msg:"R upper" (Float.abs v <= tol))
                ra);
          case classify path "svd-thin" (fun () ->
              let a = mk64 [| 4; 3 |] m4x3 in
              let u, s, vh = B.svd ~full_matrices:false a in
              let sa = F.to_array s in
              equal ~msg:"S shape" (array int) [| 3 |] (F.shape s);
              (* singular values descending and non-negative *)
              Array.iteri
                (fun i v ->
                  is_true ~msg:"S>=0" (v >= -.tol);
                  if i > 0 then
                    is_true ~msg:"S descending" (sa.(i - 1) >= v -. tol))
                sa;
              let recon =
                F.to_array (mm (mm u (mk64 [| 3; 3 |] (diag 3 sa))) vh)
              in
              is_true
                ~msg:(Printf.sprintf "USVh=A rel %g" (rel_resid m4x3 recon))
                (rok m4x3 recon));
          case classify path "eigh-symmetric" (fun () ->
              let a = mk64 [| 4; 4 |] spd4 in
              let w, v = B.eigh a in
              let wa = F.to_array w in
              (* A V = V diag(w) *)
              let av = F.to_array (mm a v) in
              let vd = F.to_array (mm v (mk64 [| 4; 4 |] (diag 4 wa))) in
              is_true
                ~msg:(Printf.sprintf "AV=VD rel %g" (rel_resid av vd))
                (rok av vd);
              let vtv = F.to_array (mm (tr v) v) in
              is_true ~msg:"VtV=I" (rok (identity 4) vtv));
          case classify path "eigvalsh-symmetric" (fun () ->
              (* the values-only path matches the eigenvalues eigh returns *)
              let a = mk64 [| 4; 4 |] spd4 in
              let w, _ = B.eigh a in
              let wa = F.to_array w and wv = F.to_array (B.eigvalsh a) in
              is_true
                ~msg:
                  (Printf.sprintf "eigvalsh=eigh vals rel %g" (rel_resid wa wv))
                (rok wa wv));
          case classify path "triangular-solve" (fun () ->
              let a =
                mk64 [| 3; 3 |] [| 2.; 0.; 0.; 1.; 3.; 0.; -1.; 2.; 4. |]
              in
              let b = mk64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
              let x =
                B.triangular_solve ~upper:false ~transpose:false
                  ~unit_diag:false a b
              in
              let recon = F.to_array (mm a x) in
              is_true
                ~msg:
                  (Printf.sprintf "Ax=b rel %g"
                     (rel_resid (F.to_array b) recon))
                (rok (F.to_array b) recon));
          case classify path "triangular-solve-vector" (fun () ->
              let a =
                mk64 [| 3; 3 |] [| 2.; 0.; 0.; 1.; 3.; 0.; -1.; 2.; 4. |]
              in
              let b = mk64 [| 3 |] [| 2.; 7.; 15. |] in
              let x =
                B.triangular_solve ~upper:false ~transpose:false
                  ~unit_diag:false a b
              in
              equal ~msg:"shape" (array int) [| 3 |] (F.shape x);
              equal ~msg:"values"
                (array (ftest ~rel:1e-12 ~abs:1e-12))
                [| 1.; 2.; 3. |] (F.to_array x);
              let ba =
                mk64 [| 2; 2; 2 |] [| 2.; 0.; 1.; 3.; 4.; 0.; -2.; 5. |]
              in
              let bb = mk64 [| 2; 2 |] [| 2.; 7.; 12.; -11. |] in
              let bx =
                B.triangular_solve ~upper:false ~transpose:false
                  ~unit_diag:false ba bb
              in
              equal ~msg:"batched shape" (array int) [| 2; 2 |] (F.shape bx);
              equal ~msg:"batched values"
                (array (ftest ~rel:1e-12 ~abs:1e-12))
                [| 1.; 2.; 3.; -1. |] (F.to_array bx));
          case classify path "qr-full" (fun () ->
              let a = mk64 [| 4; 3 |] m4x3 in
              let q, r = B.qr ~reduced:false a in
              equal ~msg:"Q shape" (array int) [| 4; 4 |] (F.shape q);
              equal ~msg:"R shape" (array int) [| 4; 3 |] (F.shape r);
              let recon = F.to_array (mm q r) in
              is_true
                ~msg:(Printf.sprintf "QR=A rel %g" (rel_resid m4x3 recon))
                (rok m4x3 recon);
              is_true ~msg:"QtQ=I" (rok (identity 4) (F.to_array (mm (tr q) q))));
          case classify path "svd-full" (fun () ->
              let a = mk64 [| 4; 3 |] m4x3 in
              let u, s, vh = B.svd ~full_matrices:true a in
              equal ~msg:"U shape" (array int) [| 4; 4 |] (F.shape u);
              equal ~msg:"S shape" (array int) [| 3 |] (F.shape s);
              equal ~msg:"Vh shape" (array int) [| 3; 3 |] (F.shape vh);
              let sa = F.to_array s in
              let sigma =
                Array.init (4 * 3) (fun i ->
                    if i / 3 = i mod 3 then sa.(i / 3) else 0.0)
              in
              let recon = F.to_array (mm (mm u (mk64 [| 4; 3 |] sigma)) vh) in
              is_true
                ~msg:(Printf.sprintf "USVh=A rel %g" (rel_resid m4x3 recon))
                (rok m4x3 recon));
          case classify path "eig-general" (fun () ->
              (* nonsymmetric real matrix; eigenpairs are complex. Residual A V
                 = V diag(w) in ℂ catches AppxA#6/#7 without a stored oracle. *)
              let n = 3 in
              let a =
                mk64 [| n; n |] [| 2.; -1.; 0.; 1.; 3.; -1.; 0.; 1.; 2. |]
              in
              let w, v = B.eig a in
              let wa = F.to_array w in
              let ac = B.cast ~dtype:F.complex128 a in
              let av = F.to_array (mm ac v) in
              let cdiag =
                Array.init (n * n) (fun i ->
                    if i / n = i mod n then wa.(i / n) else Complex.zero)
              in
              let vd =
                F.to_array (mm v (F.create ctx F.complex128 [| n; n |] cdiag))
              in
              is_true
                ~msg:(Printf.sprintf "AV=VL rel %g" (crel_resid av vd))
                (crel_resid av vd <= tol));
          case classify path "eigvals-general" (fun () ->
              (* the values-only path matches the eigenvalues eig returns *)
              let n = 3 in
              let a =
                mk64 [| n; n |] [| 2.; -1.; 0.; 1.; 3.; -1.; 0.; 1.; 2. |]
              in
              let w, _ = B.eig a in
              let wa = F.to_array w and wv = F.to_array (B.eigvals a) in
              is_true
                ~msg:
                  (Printf.sprintf "eigvals=eig vals rel %g" (crel_resid wa wv))
                (crel_resid wa wv <= tol));
          case classify path "eigh-batched" (fun () ->
              (* two stacked symmetric matrices; per-batch A V = V diag(w) *)
              let n = 3 in
              let m1 = [| 2.; 1.; 0.; 1.; 3.; 1.; 0.; 1.; 2. |] in
              let m2 = [| 4.; -1.; 0.; -1.; 4.; -1.; 0.; -1.; 4. |] in
              let a = mk64 [| 2; n; n |] (Array.append m1 m2) in
              let w, v = B.eigh a in
              equal ~msg:"w shape" (array int) [| 2; n |] (F.shape w);
              let sa = F.to_array a
              and va = F.to_array v
              and wa = F.to_array w in
              List.iter
                (fun bi ->
                  let ab = Array.sub sa (bi * n * n) (n * n) in
                  let vb = Array.sub va (bi * n * n) (n * n) in
                  let wb = Array.sub wa (bi * n) n in
                  let av = matmul_ref2 n n n ab vb in
                  let vd = matmul_ref2 n n n vb (diag n wb) in
                  is_true
                    ~msg:
                      (Printf.sprintf "batch %d AV=VD rel %g" bi
                         (rel_resid av vd))
                    (rok av vd))
                [ 0; 1 ]);
          case classify path "cholesky-complex" (fun () ->
              (* Hermitian positive-definite complex matrix: A = L L^H *)
              let z re im = { Complex.re; im } in
              let n = 2 in
              let a_arr = [| z 4. 0.; z 1. 1.; z 1. (-1.); z 3. 0. |] in
              let a = F.create ctx F.complex128 [| n; n |] a_arr in
              let l = B.cholesky ~upper:false a in
              let la = F.to_array l in
              let lh =
                Array.init (n * n) (fun idx ->
                    let i = idx / n and j = idx mod n in
                    Complex.conj la.((j * n) + i))
              in
              let recon =
                F.to_array (mm l (F.create ctx F.complex128 [| n; n |] lh))
              in
              is_true
                ~msg:(Printf.sprintf "LL^H=A rel %g" (crel_resid a_arr recon))
                (crel_resid a_arr recon <= tol));
        ];
    ]

  (* ───── Complex ───── *)

  type cdt = CDT : (Complex.t, 'b) Dtype.t * string * float -> cdt

  let cpool =
    Array.init 18 (fun i ->
        {
          Complex.re = float_of_int ((i mod 5) - 2) +. 0.5;
          im = float_of_int ((i mod 3) - 1) -. 0.25;
        })

  let complex_dtypes =
    [ CDT (F.complex64, "c32", 1e-5); CDT (F.complex128, "c64", 1e-11) ]

  let complex_tests classify path =
    [
      group "complex"
        (List.map
           (fun (CDT (dt, name, tol)) ->
             let ct = ctest ~rel:tol ~abs:tol in
             case classify path name (fun () ->
                 let a = F.create ctx dt [| 3; 4 |] (Array.sub cpool 0 12) in
                 let b = F.create ctx dt [| 3; 4 |] (Array.sub cpool 4 12) in
                 let sa = F.to_array a and sb = F.to_array b in
                 let chk n op ref =
                   equal ~msg:n (array ct) (Array.map2 ref sa sb)
                     (F.to_array (op a b))
                 in
                 chk "add" B.add Complex.add;
                 chk "sub" B.sub Complex.sub;
                 chk "mul" B.mul Complex.mul;
                 chk "div" B.fdiv Complex.div;
                 equal ~msg:"neg" (array ct) (Array.map Complex.neg sa)
                   (F.to_array (B.neg a));
                 let bt_base =
                   F.create ctx dt [| 4; 3 |] (Array.sub cpool 4 12)
                 in
                 let bt = B.permute bt_base [| 1; 0 |] in
                 let lbt =
                   permute_arr [| 4; 3 |] [| 1; 0 |] (F.to_array bt_base)
                 in
                 equal ~msg:"mul-transposed" (array ct)
                   (Array.map2 Complex.mul sa lbt)
                   (F.to_array (B.mul a bt));
                 (* complex matmul — also underwrites the eig residual oracle *)
                 let m = 2 and k = 3 and nn = 2 in
                 let ma =
                   F.create ctx dt [| m; k |] (Array.sub cpool 0 (m * k))
                 in
                 let mb =
                   F.create ctx dt [| k; nn |] (Array.sub cpool 4 (k * nn))
                 in
                 let sma = F.to_array ma and smb = F.to_array mb in
                 let mexp =
                   Array.init (m * nn) (fun idx ->
                       let i = idx / nn and j = idx mod nn in
                       let acc = ref Complex.zero in
                       for p = 0 to k - 1 do
                         acc :=
                           Complex.add !acc
                             (Complex.mul sma.((i * k) + p) smb.((p * nn) + j))
                       done;
                       !acc)
                 in
                 equal ~msg:"matmul" (array ct) mexp
                   (F.to_array (B.matmul ma mb))))
           complex_dtypes);
    ]

  (* ───── Bool ─────

     Bool participates in logical ops, comparisons, min/max, and where. *)

  let bool_tests classify path =
    [
      group "bool"
        [
          case classify path "logical" (fun () ->
              let a =
                F.create ctx F.bool [| 4 |] [| true; true; false; false |]
              in
              let b =
                F.create ctx F.bool [| 4 |] [| true; false; true; false |]
              in
              equal ~msg:"and" (array bool)
                [| true; false; false; false |]
                (F.to_array (B.and_ a b));
              equal ~msg:"or" (array bool)
                [| true; true; true; false |]
                (F.to_array (B.or_ a b));
              equal ~msg:"xor" (array bool)
                [| false; true; true; false |]
                (F.to_array (B.xor a b));
              equal ~msg:"max=or" (array bool)
                [| true; true; true; false |]
                (F.to_array (B.max a b));
              equal ~msg:"min=and" (array bool)
                [| true; false; false; false |]
                (F.to_array (B.min a b));
              equal ~msg:"cmpeq" (array bool)
                [| true; false; false; true |]
                (F.to_array (B.cmpeq a b)));
          case classify path "where" (fun () ->
              let c =
                F.create ctx F.bool [| 4 |] [| true; false; true; false |]
              in
              let x = F.create ctx F.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
              let y = F.create ctx F.int32 [| 4 |] [| 10l; 20l; 30l; 40l |] in
              equal ~msg:"select" (array int32) [| 1l; 20l; 3l; 40l |]
                (F.to_array (B.where c x y)));
          case classify path "reduce-min" (fun () ->
              let t =
                F.create ctx F.bool [| 2; 3 |]
                  [| true; true; false; true; true; true |]
              in
              equal ~msg:"bool min is all" (array bool) [| false; true |]
                (F.to_array (B.reduce ~op:`Min ~axes:[| 1 |] t)));
        ];
    ]

  (* ───── Appendix-A regressions (pointed cases) ───── *)

  let regression_tests classify path =
    let p = path in
    let ftst = ftest ~rel:0.0 ~abs:0.0 in
    [
      group "regression"
        [
          case classify p "int-div-by-zero" (fun () ->
              (* Integer division by zero returns 0. *)
              let a = F.create ctx F.int32 [| 3 |] [| 7l; -6l; 0l |] in
              let z = F.create ctx F.int32 [| 3 |] [| 0l; 0l; 0l |] in
              equal ~msg:"div0" (array int32) [| 0l; 0l; 0l |]
                (F.to_array (B.idiv a z)));
          case classify p "int-recip-by-zero" (fun () ->
              (* Integer reciprocal of zero returns 0. *)
              let t = F.create ctx F.int32 [| 3 |] [| 0l; 2l; 1l |] in
              equal ~msg:"recip0" (array int32) [| 0l; 0l; 1l |]
                (F.to_array (B.recip t)));
          case classify p "int-mod-by-zero" (fun () ->
              let a = F.create ctx F.int32 [| 3 |] [| 7l; -6l; 5l |] in
              let z = F.create ctx F.int32 [| 3 |] [| 0l; 0l; 0l |] in
              equal ~msg:"mod0" (array int32) [| 0l; 0l; 0l |]
                (F.to_array (B.mod_ a z)));
          case classify p "empty-axis-max" (fun () ->
              (* Min/max over an empty axis raises. *)
              let t = F.create ctx F.float64 [| 0; 3 |] [||] in
              raises_match ~msg:"invalid_arg"
                (function Invalid_argument _ -> true | _ -> false)
                (fun () -> B.reduce ~op:`Max ~axes:[| 0 |] t));
          case classify p "empty-axis-min" (fun () ->
              let t = F.create ctx F.float64 [| 0; 3 |] [||] in
              raises_match ~msg:"invalid_arg"
                (function Invalid_argument _ -> true | _ -> false)
                (fun () -> B.reduce ~op:`Min ~axes:[| 0 |] t));
          case classify p "empty-axis-argmax" (fun () ->
              let t = F.create ctx F.float64 [| 0; 3 |] [||] in
              raises_match ~msg:"invalid_arg"
                (function Invalid_argument _ -> true | _ -> false)
                (fun () -> B.argmax ~axis:0 ~keepdims:false t));
          case classify p "empty-axis-argmin" (fun () ->
              let t = F.create ctx F.float64 [| 0; 3 |] [||] in
              raises_match ~msg:"invalid_arg"
                (function Invalid_argument _ -> true | _ -> false)
                (fun () -> B.argmin ~axis:0 ~keepdims:false t));
          case classify p "nan-reduce-max" (fun () ->
              (* Reduce max/min propagate NaN. *)
              let t =
                F.create ctx F.float64 [| 4 |] [| 1.; Float.nan; 3.; 2. |]
              in
              is_true ~msg:"max propagates NaN"
                (Float.is_nan
                   (F.to_array (B.reduce ~op:`Max ~axes:[| 0 |] t)).(0)));
          case classify p "nan-reduce-min" (fun () ->
              let t =
                F.create ctx F.float64 [| 4 |] [| 1.; Float.nan; 3.; 2. |]
              in
              is_true ~msg:"min propagates NaN"
                (Float.is_nan
                   (F.to_array (B.reduce ~op:`Min ~axes:[| 0 |] t)).(0)));
          case classify p "argmax-nan-first" (fun () ->
              (* NaN wins, with the first index. *)
              let t = F.create ctx F.float64 [| 3 |] [| Float.nan; 1.; 2. |] in
              equal ~msg:"nan idx 0" (array int32) [| 0l |]
                (F.to_array (B.argmax ~axis:0 ~keepdims:false t)));
          case classify p "argmin-nan-first" (fun () ->
              let t = F.create ctx F.float64 [| 3 |] [| Float.nan; 1.; 2. |] in
              equal ~msg:"nan idx 0" (array int32) [| 0l |]
                (F.to_array (B.argmin ~axis:0 ~keepdims:false t)));
          case classify p "nan-sort-ascending" (fun () ->
              let t =
                F.create ctx F.float64 [| 5 |]
                  [| 1.; Float.nan; 2.; Float.nan; 0. |]
              in
              equal ~msg:"nan last" (array ftst)
                [| 0.; 1.; 2.; Float.nan; Float.nan |]
                (F.to_array (B.sort ~axis:0 ~descending:false t)));
          case classify p "nan-sort-descending" (fun () ->
              let t =
                F.create ctx F.float64 [| 5 |]
                  [| 1.; Float.nan; 2.; Float.nan; 0. |]
              in
              equal ~msg:"nan last" (array ftst)
                [| 2.; 1.; 0.; Float.nan; Float.nan |]
                (F.to_array (B.sort ~axis:0 ~descending:true t)));
          case classify p "complex-nan-sort" (fun () ->
              (* Complex sort places NaNs last. *)
              let z re im = { Complex.re; im } in
              let ct = ctest ~rel:0.0 ~abs:0.0 in
              let t =
                F.create ctx F.complex64 [| 3 |]
                  [| z 1. 0.; z Float.nan 0.; z 2. 0. |]
              in
              equal ~msg:"nan last" (array ct)
                [| z 1. 0.; z 2. 0.; z Float.nan 0. |]
                (F.to_array (B.sort ~axis:0 ~descending:false t)));
          case classify p "complex-lexicographic-sort" (fun () ->
              (* Complex sort is lexicographic (real, then imaginary), not by
                 magnitude. These inputs distinguish the two orders. *)
              let z re im = { Complex.re; im } in
              let ct = ctest ~rel:0.0 ~abs:0.0 in
              let t =
                F.create ctx F.complex64 [| 4 |]
                  [| z 2. 1.; z 1. 3.; z 2. (-1.); z 1. 0. |]
              in
              equal ~msg:"lex asc" (array ct)
                [| z 1. 0.; z 1. 3.; z 2. (-1.); z 2. 1. |]
                (F.to_array (B.sort ~axis:0 ~descending:false t)));
          case classify p "complex-sort-ties-nan" (fun () ->
              (* 8 elements with lexicographic ties (two 1+3i) and NaN in real
                 only, imag only, and both. An element with NaN in EITHER
                 component is placed last; the non-NaN prefix is lexicographic.
                 The order WITHIN the NaN group is unspecified, so we assert the
                 prefix exactly and only that the suffix is entirely NaN. *)
              let z re im = { Complex.re; im } in
              let nan = Float.nan in
              let is_cnan w =
                Float.is_nan w.Complex.re || Float.is_nan w.Complex.im
              in
              let ct = ctest ~rel:0.0 ~abs:0.0 in
              let input =
                [|
                  z 2. 1.;
                  z 1. 3.;
                  z 2. (-1.);
                  z 1. 0.;
                  z nan 5.;
                  z 3. nan;
                  z 1. 3.;
                  z nan nan;
                |]
              in
              let t = F.create ctx F.complex64 [| 8 |] input in
              let got = F.to_array (B.sort ~axis:0 ~descending:false t) in
              let sorted_finite =
                List.filter (fun w -> not (is_cnan w)) (Array.to_list input)
                |> List.sort (fun a b ->
                    let c = compare a.Complex.re b.Complex.re in
                    if c <> 0 then c else compare a.Complex.im b.Complex.im)
              in
              List.iteri
                (fun i w ->
                  is_true
                    ~msg:(Printf.sprintf "lex[%d]" i)
                    (Testable.equal ct w got.(i)))
                sorted_finite;
              for i = List.length sorted_finite to Array.length got - 1 do
                is_true
                  ~msg:(Printf.sprintf "nan suffix[%d]" i)
                  (is_cnan got.(i))
              done);
          case classify p "complex-mod-rejected" (fun () ->
              (* Complex has no mod; it must be rejected, not identity. *)
              let z re im = { Complex.re; im } in
              let a = F.create ctx F.complex64 [| 2 |] [| z 1. 1.; z 2. 0. |] in
              let b = F.create ctx F.complex64 [| 2 |] [| z 1. 0.; z 1. 1. |] in
              raises_match ~msg:"complex mod rejected"
                (fun _ -> true)
                (fun () -> B.mod_ a b));
          case classify p "complex-cmplt-rejected" (fun () ->
              (* Complex has no ordered comparison. *)
              let z re im = { Complex.re; im } in
              let a = F.create ctx F.complex64 [| 2 |] [| z 1. 1.; z 2. 0. |] in
              let b = F.create ctx F.complex64 [| 2 |] [| z 1. 0.; z 1. 1. |] in
              raises_match ~msg:"complex cmplt rejected"
                (fun _ -> true)
                (fun () -> B.cmplt a b));
          case classify p "complex-round-rejected" (fun () ->
              (* Rounding ops on complex are rejected, not identity. *)
              let z re im = { Complex.re; im } in
              let a =
                F.create ctx F.complex64 [| 2 |] [| z 1.4 0.6; z 2.5 (-1.5) |]
              in
              raises_match ~msg:"complex round rejected"
                (fun _ -> true)
                (fun () -> B.round a));
          case classify p "f32-sum-multi-accumulator" (fun () ->
              (* F32 sums use multi-accumulator unrolling. Same witness as the
                 f16/bf16 gates, one binade up: a single f32 accumulator sums
                 2^25 ones but sticks at 2^24 (ulp 2 there, +1 rounds to even),
                 a 50% error. Any width>=2 accumulator (SIMD lanes, unrolling)
                 keeps each partial < 2^24 and reaches 2^25 exactly. This is
                 order-independent and needs no reference — unlike a random-data
                 accuracy test, whose ~5e6 sum stays under 2^24 and so cannot
                 distinguish a single accumulator from a multi-accumulator.
                 Built with [full] so no large host array is materialized. *)
              let n = 1 lsl 25 in
              let t = F.full ctx F.float32 [| n |] 1.0 in
              let got = (F.to_array (B.reduce ~op:`Sum ~axes:[| 0 |] t)).(0) in
              equal ~msg:"2^25 ones" (ftest ~rel:1e-3 ~abs:0.0) (float_of_int n)
                got);
          case classify p "f16-sum-accumulates-in-float" (fun () ->
              (* F16/bf16/fp8 reductions accumulate in float. This is the one
                 interface-visible form of AppxA#15: native-int sums are
                 wrap-invariant (two's-complement add is associative mod 2^n, so
                 a storage-width accumulator and a 64-bit one truncate
                 identically), so only the converted dtypes can witness the
                 accumulator width. An f16-width accumulator sticks at 2048 (ulp
                 2 there; 2048+1 rounds to even), so reaching 4096 requires a
                 float accumulator. *)
              let t = F.full ctx F.float16 [| 4096 |] 1.0 in
              let got = (F.to_array (B.reduce ~op:`Sum ~axes:[| 0 |] t)).(0) in
              equal ~msg:"f16 sum of 4096 ones" (ftest ~rel:0.0 ~abs:0.0) 4096.0
                got);
          case classify p "bf16-sum-accumulates-in-float" (fun () ->
              (* A bf16-width accumulator sticks at 256 (ulp 2 there). *)
              let t = F.full ctx F.bfloat16 [| 512 |] 1.0 in
              let got = (F.to_array (B.reduce ~op:`Sum ~axes:[| 0 |] t)).(0) in
              equal ~msg:"bf16 sum of 512 ones" (ftest ~rel:0.0 ~abs:0.0) 512.0
                got);
          case classify p "complex-abs" (fun () ->
              (* Complex abs = |z| + 0i (never imag-zeroing of z itself). *)
              let z re im = { Complex.re; im } in
              let ct = ctest ~rel:1e-5 ~abs:1e-5 in
              let a =
                F.create ctx F.complex64 [| 3 |]
                  [| z 3. 4.; z (-1.) 2.; z 0. 0. |]
              in
              let cabs w = z (Float.hypot w.Complex.re w.Complex.im) 0.0 in
              equal ~msg:"|z|+0i" (array ct)
                (Array.map cabs (F.to_array a))
                (F.to_array (B.abs a)));
          case classify p "complex-sign" (fun () ->
              (* Complex sign = z/|z|, sign 0 = 0 (an earlier backend had a NULL
                 hole raises "operation not supported"). *)
              let z re im = { Complex.re; im } in
              let ct = ctest ~rel:1e-5 ~abs:1e-5 in
              let a =
                F.create ctx F.complex64 [| 3 |]
                  [| z 0. 0.; z 3. 4.; z (-6.) 8. |]
              in
              let csign w =
                let m = Float.hypot w.Complex.re w.Complex.im in
                if m = 0.0 then Complex.zero
                else z (w.Complex.re /. m) (w.Complex.im /. m)
              in
              equal ~msg:"z/|z|" (array ct)
                (Array.map csign (F.to_array a))
                (F.to_array (B.sign a)));
        ];
    ]

  (* ───── Integer pow, neg/abs, reduce_prod/scan_prod/arg* ───── *)

  (* Integer pow: special-cased exponents and modular wrap on overflow. Signed
     dtypes exercise pow(0,0)=1, negative exponents (base 1/-1 by parity,
     |base|>1 truncates to 0), and wrap at exp = bits-1 / bits; unsigned
     exercise the non-negative-exponent wrap. *)
  let int_pow_check (IDT d) bases exps =
    let n = Array.length bases in
    let tb =
      F.create ctx d.dt [| n |]
        (Array.map (fun x -> d.of_i64 (Int64.of_int x)) bases)
    in
    let te =
      F.create ctx d.dt [| n |]
        (Array.map (fun x -> d.of_i64 (Int64.of_int x)) exps)
    in
    let sb = F.to_array tb and se = F.to_array te in
    let expected =
      Array.init n (fun i ->
          d.of_i64
            (wrap ~bits:d.bits ~signed:d.signed
               (ipow64 (d.to_i64 sb.(i)) (d.to_i64 se.(i)))))
    in
    equal ~msg:"int pow" (array d.tst) expected (F.to_array (B.pow tb te))

  let int_pow_tests classify path =
    [
      group "int_pow"
        (List.map
           (fun (IDT d as idt) ->
             (* special-value cases that overflow no dtype: 0^0=1, negative
                exponents, and small positive powers *)
             let bases, exps =
               if d.signed then
                 ([| 0; -1; -1; 1; 2; 3; 2 |], [| 0; -3; -4; -5; -1; 2; 4 |])
               else ([| 0; 1; 2; 3; 2; 1; 2 |], [| 0; 5; 3; 2; 4; 7; 4 |])
             in
             case classify path d.name (fun () -> int_pow_check idt bases exps))
           int_dtypes);
      group "int_pow_wrap"
        (List.map
           (fun (IDT d as idt) ->
             (* modular wrap on overflow: 2^(bits-1) and 2^bits across every
                integer storage width. *)
             case classify "int_pow_wrap" d.name (fun () ->
                 int_pow_check idt [| 2; 2 |] [| d.bits - 1; d.bits |]))
           int_dtypes);
    ]

  (* Integer neg / abs, with pools that include INT_MIN (signed) / the all-ones
     max (unsigned) — where signed-overflow UB hides. numpy semantics: neg and
     abs of INT_MIN wrap back to INT_MIN. *)
  let int_neg_abs_tests classify path =
    let dmin bits = Int64.neg (Int64.shift_left 1L (bits - 1)) in
    let dmax_s bits = Int64.sub (Int64.shift_left 1L (bits - 1)) 1L in
    let dmax_u bits =
      if bits >= 64 then -1L else Int64.sub (Int64.shift_left 1L bits) 1L
    in
    [
      group "int_neg_abs"
        (List.map
           (fun (IDT d) ->
             let vals =
               if d.signed then
                 [| dmin d.bits; dmax_s d.bits; 0L; 1L; -1L; 5L; -7L; 3L |]
               else [| 0L; 1L; dmax_u d.bits; 5L; 2L; 10L; 128L; 3L |]
             in
             case classify (path ^ "/int_neg_abs") d.name (fun () ->
                 let n = Array.length vals in
                 let t = F.create ctx d.dt [| n |] (Array.map d.of_i64 vals) in
                 let sv = F.to_array t in
                 let neg_exp =
                   Array.map
                     (fun v ->
                       d.of_i64
                         (o_neg ~bits:d.bits ~signed:d.signed (d.to_i64 v)))
                     sv
                 in
                 let abs_exp =
                   Array.map
                     (fun v ->
                       d.of_i64
                         (o_abs ~bits:d.bits ~signed:d.signed (d.to_i64 v)))
                     sv
                 in
                 equal ~msg:"neg" (array d.tst) neg_exp (F.to_array (B.neg t));
                 equal ~msg:"abs" (array d.tst) abs_exp (F.to_array (B.abs t))))
           int_dtypes);
    ]

  (* Integer reduce_prod / scan_prod / arg* ─────

     The generic reduce/scan/arg drivers are float-only (prod overflows the
     high-bit int pools, and arg* carry the NaN rule); these focused fixtures
     cover the integer paths, including unsigned argmax over a high-bit
     value. *)
  let int_extra_tests classify path =
    [
      group "int_extra"
        [
          case classify path "reduce-prod" (fun () ->
              let t =
                F.create ctx F.int32 [| 2; 3 |] [| 1l; 2l; 3l; 2l; 2l; 2l |]
              in
              equal ~msg:"prod axis1" (array int32) [| 6l; 8l |]
                (F.to_array (B.reduce ~op:`Prod ~axes:[| 1 |] t)));
          case classify path "cumprod" (fun () ->
              let t =
                F.create ctx F.int32 [| 2; 3 |] [| 1l; 2l; 3l; 2l; 2l; 2l |]
              in
              equal ~msg:"cumprod axis1" (array int32)
                [| 1l; 2l; 6l; 2l; 4l; 8l |]
                (F.to_array (B.associative_scan ~axis:1 ~op:`Prod t)));
          case classify path "argmax" (fun () ->
              let t =
                F.create ctx F.int32 [| 2; 3 |] [| 3l; 1l; 2l; 5l; 9l; 4l |]
              in
              equal ~msg:"argmax axis1" (array int32) [| 0l; 1l |]
                (F.to_array (B.argmax ~axis:1 ~keepdims:false t)));
          case classify path "argmin" (fun () ->
              let t =
                F.create ctx F.int32 [| 2; 3 |] [| 3l; 1l; 2l; 5l; 9l; 4l |]
              in
              equal ~msg:"argmin axis1" (array int32) [| 1l; 2l |]
                (F.to_array (B.argmin ~axis:1 ~keepdims:false t)));
          case classify path "argsort" (fun () ->
              let ta = [| 3l; 1l; 2l; 0l |] in
              let idx =
                F.to_array
                  (B.argsort ~axis:0 ~descending:false
                     (F.create ctx F.int32 [| 4 |] ta))
              in
              equal ~msg:"reconstruct" (array int32) [| 0l; 1l; 2l; 3l |]
                (Array.map (fun j -> ta.(Int32.to_int j)) idx));
          case classify path "uint32-argmax-unsigned" (fun () ->
              (* unsigned max is the high-bit value at index 1, not 5 *)
              let t = F.create ctx F.uint32 [| 3 |] [| 5l; 0x80000000l; 3l |] in
              equal ~msg:"unsigned argmax" (array int32) [| 1l |]
                (F.to_array (B.argmax ~axis:0 ~keepdims:false t)));
        ];
    ]

  (* ───── FFT ─────

     The BACKEND transforms are unnormalized (the frontend applies the 1/n for
     the "backward" norm), so B.fft is the forward DFT, B.ifft the unnormalized
     inverse DFT, and B.ifft (B.fft x) = n * x. Forward and inverse are checked
     directly against an O(n^2) DFT; the multi-axis and real round-trips are
     scaled by the transformed size. *)

  let fft_tests classify path =
    let ct = ctest ~rel:1e-9 ~abs:1e-9 in
    let z re im = { Complex.re; im } in
    let dft ~inverse (a : Complex.t array) =
      let n = Array.length a in
      let s = if inverse then 1.0 else -1.0 in
      Array.init n (fun k ->
          let acc = ref Complex.zero in
          for j = 0 to n - 1 do
            let ang =
              s *. 2.0 *. Float.pi *. float_of_int (k * j) /. float_of_int n
            in
            acc := Complex.add !acc (Complex.mul a.(j) (z (cos ang) (sin ang)))
          done;
          !acc)
    in
    let cdata n =
      Array.init n (fun i ->
          z (float_of_int ((i mod 5) - 2)) (float_of_int ((i mod 3) - 1)))
    in
    let cscale s =
      Array.map (fun w -> z (w.Complex.re *. s) (w.Complex.im *. s))
    in
    let dft_case name n =
      case classify path name (fun () ->
          let t = F.create ctx F.complex128 [| n |] (cdata n) in
          equal ~msg:"fft = DFT" (array ct)
            (dft ~inverse:false (F.to_array t))
            (F.to_array (B.fft t ~axes:[| 0 |])))
    in
    [
      group "fft"
        [
          case classify path "fft-1d-vs-dft" (fun () ->
              let n = 8 in
              let t = F.create ctx F.complex128 [| n |] (cdata n) in
              equal ~msg:"fft = DFT" (array ct)
                (dft ~inverse:false (F.to_array t))
                (F.to_array (B.fft t ~axes:[| 0 |])));
          dft_case "fft-radix7-vs-dft" 7;
          dft_case "fft-mixed-radix-vs-dft" 60;
          dft_case "fft-odd-prime-product-vs-dft" 143;
          dft_case "fft-bluestein-vs-dft" 17;
          case classify path "fft-complex64-bluestein-vs-dft" (fun () ->
              let n = 17 in
              let t = F.create ctx F.complex64 [| n |] (cdata n) in
              equal ~msg:"complex64 fft = DFT"
                (array (ctest ~rel:1e-4 ~abs:1e-4))
                (dft ~inverse:false (F.to_array t))
                (F.to_array (B.fft t ~axes:[| 0 |])));
          case classify path "fft-strided-axis-vs-dft" (fun () ->
              let n = 17 in
              let interleaved =
                Array.init (n * 2) (fun i ->
                    if i mod 2 = 0 then z 99. (-99.) else (cdata n).(i / 2))
              in
              let base = F.create ctx F.complex128 [| n; 2 |] interleaved in
              let t = B.shrink base [| (0, n); (1, 2) |] in
              equal ~msg:"strided fft = DFT" (array ct)
                (dft ~inverse:false (F.to_array t))
                (F.to_array (B.fft t ~axes:[| 0 |])));
          case classify path "ifft-1d-vs-dft" (fun () ->
              (* unnormalized inverse: B.ifft = the +sign DFT, no 1/n *)
              let n = 8 in
              let t = F.create ctx F.complex128 [| n |] (cdata n) in
              equal ~msg:"ifft = inverse DFT" (array ct)
                (dft ~inverse:true (F.to_array t))
                (F.to_array (B.ifft t ~axes:[| 0 |])));
          case classify path "ifft-fft-is-n-times" (fun () ->
              let n = 8 in
              let t = F.create ctx F.complex128 [| n |] (cdata n) in
              let rt = B.ifft (B.fft t ~axes:[| 0 |]) ~axes:[| 0 |] in
              equal ~msg:"ifft(fft x) = n x" (array ct)
                (cscale (float_of_int n) (F.to_array t))
                (F.to_array rt));
          case classify path "fft-2d-roundtrip" (fun () ->
              let t = F.create ctx F.complex128 [| 3; 4 |] (cdata 12) in
              let rt = B.ifft (B.fft t ~axes:[| 0; 1 |]) ~axes:[| 0; 1 |] in
              equal ~msg:"2d roundtrip = 12 x" (array ct)
                (cscale 12.0 (F.to_array t))
                (F.to_array rt));
          case classify path "rfft-irfft-roundtrip-even" (fun () ->
              let n = 8 in
              let data =
                Array.init n (fun i -> float_of_int ((i mod 4) - 1) +. 0.5)
              in
              let t = F.create ctx F.float64 [| n |] data in
              let spec = B.rfft t ~dtype:F.complex128 ~axes:[| 0 |] in
              let back =
                B.irfft ~s:[| n |] spec ~dtype:F.float64 ~axes:[| 0 |]
              in
              equal ~msg:"irfft(rfft x) = n x"
                (array (ftest ~rel:1e-9 ~abs:1e-9))
                (Array.map (fun v -> v *. float_of_int n) (F.to_array t))
                (F.to_array back));
          case classify path "rfft-irfft-roundtrip-odd" (fun () ->
              (* odd n exercises the half-spectrum length (n+1)/2 differently *)
              let n = 7 in
              let data =
                Array.init n (fun i -> float_of_int ((i mod 5) - 2) +. 0.25)
              in
              let t = F.create ctx F.float64 [| n |] data in
              let spec = B.rfft t ~dtype:F.complex128 ~axes:[| 0 |] in
              let back =
                B.irfft ~s:[| n |] spec ~dtype:F.float64 ~axes:[| 0 |]
              in
              equal ~msg:"irfft(rfft x) = n x"
                (array (ftest ~rel:1e-9 ~abs:1e-9))
                (Array.map (fun v -> v *. float_of_int n) (F.to_array t))
                (F.to_array back));
          case classify path "irfft-explicit-size-truncates" (fun () ->
              let one = z 1. 0. in
              let spec = F.create ctx F.complex128 [| 9 |] (Array.make 9 one) in
              let back =
                B.irfft ~s:[| 8 |] spec ~dtype:F.float64 ~axes:[| 0 |]
              in
              equal ~msg:"long half-spectrum is truncated"
                (array (ftest ~rel:1e-9 ~abs:1e-9))
                [| 8.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
                (F.to_array back));
          case classify path "irfft-explicit-size-pads" (fun () ->
              let one = z 1. 0. in
              let spec = F.create ctx F.complex128 [| 3 |] (Array.make 3 one) in
              let back =
                B.irfft ~s:[| 8 |] spec ~dtype:F.float64 ~axes:[| 0 |]
              in
              let root2 = Stdlib.sqrt 2. in
              equal ~msg:"short half-spectrum is zero-padded"
                (array (ftest ~rel:1e-9 ~abs:1e-9))
                [|
                  5.;
                  1. +. root2;
                  -1.;
                  1. -. root2;
                  1.;
                  1. -. root2;
                  -1.;
                  1. +. root2;
                |]
                (F.to_array back));
          case classify path "fft-known-value" (fun () ->
              (* hand-computed: fft([1,1,1,1]) = [4,0,0,0]; fft of a unit delta
                 [1,0,0,0] = [1,1,1,1] (all ones). *)
              let ones =
                F.create ctx F.complex128 [| 4 |] (Array.make 4 (z 1. 0.))
              in
              equal ~msg:"fft(1,1,1,1)" (array ct)
                [| z 4. 0.; z 0. 0.; z 0. 0.; z 0. 0. |]
                (F.to_array (B.fft ones ~axes:[| 0 |]));
              let delta =
                F.create ctx F.complex128 [| 4 |]
                  [| z 1. 0.; z 0. 0.; z 0. 0.; z 0. 0. |]
              in
              equal ~msg:"fft(delta)" (array ct)
                (Array.make 4 (z 1. 0.))
                (F.to_array (B.fft delta ~axes:[| 0 |])));
          case classify path "parseval" (fun () ->
              (* unnormalized DFT: sum |X_k|^2 = n * sum |x_j|^2 *)
              let n = 8 in
              let t = F.create ctx F.complex128 [| n |] (cdata n) in
              let energy a =
                Array.fold_left
                  (fun s w ->
                    s
                    +. (w.Complex.re *. w.Complex.re)
                    +. (w.Complex.im *. w.Complex.im))
                  0.0 a
              in
              let ex = energy (F.to_array t) in
              let ex_hat = energy (F.to_array (B.fft t ~axes:[| 0 |])) in
              is_true
                ~msg:
                  (Printf.sprintf "Parseval %g vs %g" ex_hat
                     (float_of_int n *. ex))
                (Float.abs (ex_hat -. (float_of_int n *. ex))
                <= 1e-6 *. float_of_int n *. ex));
        ];
    ]

  (* ───── High rank (MAX_NDIM boundary) ─────

     NX_C_MAX_NDIM is 32 and requires a registered test at that rank. Shape is
     32 dims, all 1 except the two leading 2s (4 elements), so the boundary is
     exercised without a large tensor. *)

  let highrank_tests classify path =
    let ftst = ftest ~rel:0.0 ~abs:0.0 in
    let nd = 32 in
    let shape = Array.init nd (fun i -> if i < 2 then 2 else 1) in
    let data = [| 1.; 2.; 3.; 4. |] in
    let mk () = F.create ctx F.float64 shape data in
    [
      group "highrank"
        [
          case classify path "rank32-add" (fun () ->
              let t = mk () in
              equal ~msg:"shape" (array int) shape (F.shape t);
              equal ~msg:"add self" (array ftst) [| 2.; 4.; 6.; 8. |]
                (F.to_array (B.add t t)));
          case classify path "rank32-reduce" (fun () ->
              let got = B.reduce ~op:`Sum ~axes:[| 0 |] (mk ()) in
              equal ~msg:"shape" (array int)
                (Array.sub shape 1 (nd - 1))
                (F.shape got);
              equal ~msg:"sum axis0" (array ftst) [| 4.; 6. |] (F.to_array got));
          case classify path "rank32-permute" (fun () ->
              let axes =
                Array.init nd (fun i ->
                    if i = 0 then 1 else if i = 1 then 0 else i)
              in
              let p = B.permute (mk ()) axes in
              equal ~msg:"shape" (array int) shape (F.shape p);
              equal ~msg:"swapped leading axes" (array ftst)
                [| 1.; 3.; 2.; 4. |] (F.to_array p));
        ];
    ]

  (* ───── fp8 elementwise ─────

     fp8e4m3/e5m2 are converted-compute (storage != compute; the LOAD/STORE
     table converts to/from float), so their arithmetic path is exactly where a
     converter table row hides a bug that cast-only coverage misses. Values sit
     well inside both formats' ranges; tolerances are loose because a single
     round to fp8 costs up to ~2^-3 (e4m3) / ~2^-2 (e5m2) relative. *)

  let fp8pool =
    [|
      1.0;
      2.0;
      -1.0;
      0.5;
      -2.0;
      3.0;
      -0.5;
      4.0;
      1.5;
      -3.0;
      2.5;
      -1.5;
      0.75;
      -4.0;
      3.5;
      -2.5;
      1.25;
      -0.75;
    |]

  let fp8_dtypes =
    [
      FDT
        {
          dt = F.float8_e4m3;
          name = "fp8e4m3";
          pool = fp8pool;
          a_rel = 0.15;
          a_abs = 0.1;
          t_rel = 0.3;
          t_abs = 0.2;
        };
      FDT
        {
          dt = F.float8_e5m2;
          name = "fp8e5m2";
          pool = fp8pool;
          a_rel = 0.3;
          a_abs = 0.15;
          t_rel = 0.5;
          t_abs = 0.3;
        };
    ]

  let fp8_tests classify path =
    [
      group "fp8"
        (List.map
           (fun (FDT d) ->
             let tst = ftest ~rel:d.a_rel ~abs:d.a_abs in
             case classify path d.name (fun () ->
                 List.iter
                   (fun (lname, ta, tb, la, lb) ->
                     equal ~msg:("add " ^ lname) (array tst)
                       (Array.map2 ( +. ) la lb)
                       (F.to_array (B.add ta tb));
                     equal ~msg:("mul " ^ lname) (array tst)
                       (Array.map2 ( *. ) la lb)
                       (F.to_array (B.mul ta tb)))
                   (float_bin_cases d.dt d.pool d.pool);
                 List.iter
                   (fun (lname, t, logical, _) ->
                     equal ~msg:("neg " ^ lname) (array tst)
                       (Array.map (fun x -> -.x) logical)
                       (F.to_array (B.neg t)))
                   (List.filter
                      (fun (n, _, _, _) -> n = "contig" || n = "transpose")
                      (layouts d.dt d.pool))))
           fp8_dtypes);
    ]

  (* ───── Suite assembly ───── *)

  let suite () =
    let classify = Fun.const Pass in
    (* Families that own their own top group are spliced; families that return
       sub-groups/cases are wrapped here once. Every leaf test's classify key is
       the "/"-joined group path (e.g. "reduce/sum/u32"). *)
    let groups =
      List.concat
        [
          [
            group "elementwise"
              (unary_float_tests classify "elementwise"
              @ binary_float_tests classify "elementwise"
              @ int_binary_tests classify "elementwise"
              @ comparison_tests classify "elementwise");
          ];
          [
            group "reduce"
              (reduction_tests classify "reduce"
              @ argreduce_tests classify "reduce");
          ];
          [ group "scan" (scan_tests classify "scan") ];
          [ group "sort" (sort_tests classify "sort") ];
          [ group "ternary" (where_tests classify "ternary") ];
          [
            group "indexed"
              (gather_tests classify "indexed"
              @ scatter_tests classify "indexed");
          ];
          movement_tests classify "movement";
          cast_tests classify "cast";
          threefry_tests classify "threefry";
          unfold_tests classify "window";
          matmul_tests classify "matmul";
          linalg_tests classify "linalg";
          fft_tests classify "fft";
          complex_tests classify "complex";
          bool_tests classify "bool";
          int_extra_tests classify "int_extra";
          int_pow_tests classify "int_pow";
          int_neg_abs_tests classify "int_neg_abs";
          fp8_tests classify "fp8";
          highrank_tests classify "highrank";
          regression_tests classify "regression";
        ]
    in
    groups
end
