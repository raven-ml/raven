(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let devices =
  Array.init 4 (fun k ->
      Result.get_ok (Rig.memory_device (Printf.sprintf "m%d" k)))

let memory k = devices.(k)

module Counting = struct
  include Nx_cpu

  let name = "nx.test"
  let count = Atomic.make 0
  let computed = Atomic.make 0
  let calls () = Atomic.get count
  let multiplied = Atomic.make 0
  let positions () = Atomic.get computed
  let products () = Atomic.get multiplied

  let reset () =
    Atomic.set count 0;
    Atomic.set computed 0;
    Atomic.set multiplied 0

  let size shape = Array.fold_left ( * ) 1 shape

  let loop (Nx_array.Any dst) = Nx_array.Layout.numel (Nx_array.layout dst)

  let map p ~dsts loads =
    Atomic.incr count;
    ignore (Atomic.fetch_and_add computed (loop dsts.(0)));
    Nx_cpu.map p ~dsts loads

  let reduce p ~shape ~axes rs ~dsts loads =
    Atomic.incr count;
    ignore (Atomic.fetch_and_add computed (size shape));
    Nx_cpu.reduce p ~shape ~axes rs ~dsts loads

  let scan ?carry p ~axis r ~dsts loads =
    Atomic.incr count;
    ignore (Atomic.fetch_and_add computed (loop dsts.(0)));
    Nx_cpu.scan ?carry p ~axis r ~dsts loads

  let gather ~axis ~dst idx x =
    Atomic.incr count;
    Nx_cpu.gather ~axis ~dst idx x

  let scatter c ~unique ~axis ~dst ~into idx u =
    Atomic.incr count;
    Nx_cpu.scatter c ~unique ~axis ~dst ~into idx u

  let sort ~axis ~descending ~values ~positions x =
    Atomic.incr count;
    Nx_cpu.sort ~axis ~descending ~values ~positions x

  let assemble ~fill ~dst pieces =
    Atomic.incr count;
    Nx_cpu.assemble ~fill ~dst pieces

  let fold p ~dst x =
    Atomic.incr count;
    Nx_cpu.fold p ~dst x

  let contract ~batch ~contracting ~segments ~acc ~dst ~init
      (a : Nx_kernel.operand) b =
    Atomic.incr count;
    let (Nx_array.Any x) = a.array in
    let s = Nx_array.Layout.shape (Nx_array.layout x) in
    let k =
      Iarray.fold_left (fun k (i, _) -> k * Iarray.get s i) 1 contracting
    in
    ignore (Atomic.fetch_and_add multiplied (loop (Nx_array.Any dst) * k));
    Nx_cpu.contract ~batch ~contracting ~segments ~acc ~dst ~init a b
end

module Declining = struct
  include Nx_cpu

  let name = "nx.test"

  let map p ~dsts loads =
    match Nx_kernel.Prog.single p with
    | Some (Const _ | Op2 (Binary Add, _, _)) -> Nx_array.Declined
    | Some _ | None -> Nx_cpu.map p ~dsts loads
end

(* nx.cpu's one-node maps of a cast and of a sum. *)
let cast ~dst x = Nx_kernels_support.map1 (module Nx_cpu) Cast ~dst x

let sum ~dst x y =
  Nx_kernels_support.map2 (module Nx_cpu) (Binary Add) ~dst x y

module Blocked = struct
  include Nx_cpu

  let name = "nx.blocks"
  let block = 4096
  let shape x = Nx_array.Layout.shape (Nx_array.layout x)

  let ok what = function
    | Nx_array.Done -> ()
    | _ -> failwith ("nx.blocks: nx.cpu refused a " ^ what)

  (* [x]'s elements [lo, hi) along [axis]. *)
  let slice axis lo hi x =
    let r =
      Array.map
        (fun n -> { Nx_array.Move.start = 0; count = n; step = 1 })
        (Iarray.to_array (shape x))
    in
    r.(axis) <- { start = lo; count = hi - lo; step = 1 };
    Option.get (Nx_array.move (Slice (Iarray.of_array r)) x)

  (* A scaled or segmented contraction goes to nx.cpu whole. *)
  let contract ~batch ~contracting ~segments ~acc ~dst ~init
      (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
    if Iarray.length contracting > 1 then
      failwith "nx.blocks: a contraction over several pairs";
    let (Nx_array.Any a) = oa.array in
    let (Nx_array.Any b) = ob.array in
    let k =
      Iarray.fold_left
        (fun k (i, _) -> k * Iarray.get (shape a) i)
        1 contracting
    in
    let whole =
      k = 0 || Option.is_some segments || Option.is_some oa.scale
      || Option.is_some ob.scale
    in
    if whole then
      Nx_cpu.contract ~batch ~contracting ~segments ~acc ~dst ~init oa ob
    else begin
      let ia, ib = Iarray.get contracting 0 in
      let (Nx_array.Dtype.Any ad) = acc in
      let fresh () = Nx_array.create (Nx_array.device dst) ad (shape dst) in
      let part i =
        let lo = i * block and hi = Int.min k ((i + 1) * block) in
        let y = fresh () in
        ok "contraction"
          (Nx_cpu.contract ~batch ~contracting ~segments:None
             ~acc ~dst:y ~init:None
             { array = Nx_array.Any (slice ia lo hi a); scale = None }
             { array = Nx_array.Any (slice ib lo hi b); scale = None });
        y
      in
      let add x y =
        let z = fresh () in
        ok "sum" (sum ~dst:z x y);
        z
      in
      let parts = Array.init (((k - 1) / block) + 1) part in
      let n = Array.length parts in
      let init =
        Option.map
          (fun i ->
            let z = fresh () in
            ok "cast" (cast ~dst:z i);
            z)
          init
      in
      let from i = add i parts.(0) in
      let s = ref (Option.fold ~none:parts.(0) ~some:from init) in
      for i = 1 to n - 1 do
        s := add !s parts.(i)
      done;
      cast ~dst !s
    end
end

(* NaN bits *)

(* The bytes of [dt]'s floats, or of its parts, and their exponent and
   fraction widths, where a NaN has a quiet bit to set or clear:
   float8_e4m3fn's one NaN of each sign has every fraction bit set, and
   float4_e2m1fn has none. *)
let fields (type v s) (dt : (v, s) Nx_array.Dtype.t) =
  match dt with
  | Float64 | Complex128 -> Some (8, 11, 52)
  | Float32 | Complex64 -> Some (4, 8, 23)
  | Float16 -> Some (2, 5, 10)
  | Bfloat16 -> Some (2, 8, 7)
  | Float8_e5m2 -> Some (1, 5, 2)
  | _ -> None

(* [a], C-contiguous, with [f frac v] for the bits [v] of each float or part
   that is a NaN, [frac] its fraction's width. *)
let map_nans f (type v s) (a : (v, s) Nx_array.t) : (v, s) Nx_array.t =
  let a = Nx_array.copy a in
  let dt = Nx_array.dtype a in
  match fields dt with
  | None -> a
  | Some (w, exp, frac) ->
      let b = Nx_array.to_array (Option.get (Nx_array.bitcast Uint8 a)) in
      let ones = Int64.(sub (shift_left 1L exp) 1L) in
      for p = 0 to (Array.length b / w) - 1 do
        let v = ref 0L in
        for j = w - 1 downto 0 do
          v := Int64.(logor (shift_left !v 8) (of_int b.((p * w) + j)))
        done;
        let field = Int64.(logand (shift_right_logical !v frac) ones) in
        let fraction = Int64.(logand !v (sub (shift_left 1L frac) 1L)) in
        if field = ones && fraction <> 0L then begin
          let v = f frac !v in
          for j = 0 to w - 1 do
            b.((p * w) + j) <-
              Int64.to_int (Int64.shift_right_logical v (8 * j)) land 0xFF
          done
        end
      done;
      let bytes = String.init (Array.length b) (fun i -> Char.chr b.(i)) in
      Nx_array.v dt
        (Nx_array.Layout.contiguous (Nx_array.Layout.shape (Nx_array.layout a)))
        (Rig.Buffer.of_string (if bytes = "" then "\000" else bytes))

let quiet_bit frac = Int64.shift_left 1L (frac - 1)

let signalled a =
  map_nans
    (fun frac v ->
      let q = quiet_bit frac in
      if Int64.(logand v (sub q 1L)) = 0L then v
      else Int64.(logand v (lognot q)))
    a

let quieted a = map_nans (fun frac v -> Int64.logor v (quiet_bit frac)) a

let signalling (type v s) (dt : (v, s) Nx_array.Dtype.t) s (data : v array) =
  Nx.Repr.of_array Nx.Host.v
    (signalled (Nx_array.of_array dt (Iarray.of_array s) data))

let signals (type v s) (dt : (v, s) Nx_array.Dtype.t) (data : v array) =
  match Nx_array.Dtype.kind dt with
  | (Float | Complex) when Nx_array.Dtype.bits dt >= 8 ->
      let bytes a = Nx_array.to_array (Option.get (Nx_array.bitcast Uint8 a)) in
      let a = signalled (Nx_array.of_array dt [| Array.length data |] data) in
      bytes a <> bytes (quieted a)
  | _ -> false

(* Allocation *)

let words f =
  ignore (f ());
  let n = 1000 in
  let before = Gc.minor_words () in
  for _ = 1 to n do
    ignore (Sys.opaque_identity (f ()))
  done;
  (Gc.minor_words () -. before) /. Float.of_int n

(* Backtraces *)

let raised_in f =
  let recording = Printexc.backtrace_status () in
  Printexc.record_backtrace true;
  Fun.protect ~finally:(fun () -> Printexc.record_backtrace recording)
  @@ fun () ->
  match f () with
  | _ -> "no exception"
  | exception _ -> (
      let bt = Printexc.get_raw_backtrace () in
      match Printexc.backtrace_slots bt with
      | None | Some [||] -> "no backtrace"
      | Some slots -> (
          match Printexc.Slot.location slots.(0) with
          | Some l -> Filename.basename l.filename
          | None -> "no location"))

(* Witnesses and indices *)

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let index s k =
  let idx = Array.make (Array.length s) 0 and k = ref k in
  for a = Array.length s - 1 downto 0 do
    idx.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  idx

let placement () =
  Windtrap.Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

let same_float a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let same (type v s) (dt : (v, s) Nx_array.Dtype.t) (a : v) (b : v) =
  match Nx_array.Dtype.kind dt with
  | Float -> same_float a b
  | Complex -> same_float a.Complex.re b.Complex.re && same_float a.im b.im
  | Signed | Unsigned | Boolean -> a = b

let elements dt =
  Windtrap.Testable.make
    ~pp:(Format.pp_print_list (Nx_array.Dtype.pp_value dt))
    ~equal:(fun a b ->
      List.length a = List.length b && List.for_all2 (same dt) a b)

let equal_elements dt a b =
  Windtrap.equal (elements dt) (Array.to_list a) (Array.to_list b)
