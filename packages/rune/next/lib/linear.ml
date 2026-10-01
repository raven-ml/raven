(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Repr = Nx.Repr

(* A tape's entries, one per slot. A linear call's outputs are consecutive
   slots: the first holds the call, the others [Part]. *)
type entry =
  | Input
  | Recorded : ('a, 'b) Nx.t Nx.Op.t -> entry
  | Call of {
      inputs : Nx.packed list;
      like : Nx.packed list;
      pullback : Nx.packed list -> Nx.packed list;
    }
  | Part

type tape = {
  entry : string;
  mutable entries : entry array;
  mutable length : int;
}

type (_, _) Repr.node +=
  | Slot : { tape : tape; index : int } -> ('a, 'b) Repr.node

let create entry = { entry; entries = Array.make 64 Input; length = 0 }

let owns t x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Slot { tape; _ } -> tape == t
      | _ -> false)
  | Host _ | Placed _ -> false

let index x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Slot { index; _ } -> index
      | _ -> assert false)
  | Host _ | Placed _ -> assert false

let slot t ~context placement dtype shape e =
  if t.length = Array.length t.entries then begin
    let entries = Array.make (2 * t.length) Input in
    Array.blit t.entries 0 entries 0 t.length;
    t.entries <- entries
  end;
  let index = t.length in
  t.entries.(index) <- e;
  t.length <- index + 1;
  Repr.Traced.v ~context placement dtype shape (Slot { tape = t; index })

let input t x =
  slot t ~context:(Repr.context x) (Nx.placement x) (Nx.dtype x) (Nx.shape x)
    Input

let call t inputs pullback like =
  List.mapi
    (fun j (Nx.P y) ->
      let e = if j = 0 then Call { inputs; like; pullback } else Part in
      Nx.P
        (slot t ~context:(Repr.context y) (Nx.placement y) (Nx.dtype y)
           (Nx.shape y) e))
    like

(* Recording *)

let nonlinear t op =
  invalid_arg
    (Printf.sprintf
       "%s: a custom_jvp tangent map applies %s to a tangent; a tangent map \
        must be linear in its tangents"
       t.entry (name op))

(* [record t op x] is the slot of [op]'s result, recorded on [t]; [x] is a slot
   operand, whose context the result takes. *)
let record t op x =
  slot t ~context:(Repr.context x) (placement op) (Nx.Op.dtype op)
    (Nx.Op.shape op) (Recorded op)

let claims : type r. tape -> r Nx.Op.t -> bool =
 fun t op ->
  match[@warning "@4@8"] op with
  | Unary (_, x) -> owns t x
  | Binary (_, a, b) -> owns t a || owns t b
  | Reduce (_, _, x) -> owns t x
  | Move (x, _) -> owns t x
  | Matmul (a, b) -> owns t a || owns t b
  | Compare _ | Where _ | Scan _ | Arg_reduce _ | Sort _ | Argsort _ | Pad _
  | Cat _ | Convert _ | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _
  | Fold _ | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _
  | Svd _ | Eig _ | Eigh _ | Solve_triangular _ | Place _ | Read _ ->
      List.exists (fun (Nx.P x) -> owns t x) (operands op)

let real_or_complex dt = Nx_dtype.is_float dt || Nx_dtype.is_complex dt

(* [record_any t op] is [record t op x] for the first slot operand [x]. *)
let record_any t op =
  let (Nx.P x) = List.find (fun (Nx.P x) -> owns t x) (operands op) in
  record t op x

let sum t op (k : Nx_backend.reduce) x =
  match[@warning "@4@8"] k with
  | Sum -> record t op x
  | Prod | Max | Min -> nonlinear t op

(* [run t op] is [op], one of whose operands is a slot of [t]: the slot of its
   result if [op] is linear in its slots. A plain operand of an operation linear
   in several is taken as zero, as a tangent's zero fill is. *)
let run : type r. tape -> r Nx.Op.t -> r =
 fun t op ->
  match[@warning "@4@8"] op with
  | Unary (k, x) -> (
      match[@warning "@4@8"] (k : Nx_backend.unary) with
      | Neg -> record t op x
      | Recip | Abs | Sqrt | Sign | Exp | Log | Sin | Cos | Tan | Asin | Acos
      | Atan | Sinh | Cosh | Tanh | Trunc | Ceil | Floor | Round | Erf ->
          nonlinear t op)
  | Binary (k, a, b) ->
      let sa = owns t a and sb = owns t b in
      let linear =
        match[@warning "@4@8"] (k : Nx_backend.binary) with
        | Add | Sub -> sa && sb
        | Mul -> sa <> sb
        | Fdiv -> not sb
        | Idiv | Mod | Pow | Atan2 | Maximum | Minimum | And | Or | Xor -> false
      in
      if linear then record t op (if sa then a else b) else nonlinear t op
  | Where _ -> record_any t op
  | Reduce (k, _, x) -> sum t op k x
  | Scan (k, _, x) -> sum t op k x
  | Pad (_, v, x) ->
      if v = Nx_dtype.zero (Nx.dtype x) then record t op x else nonlinear t op
  | Convert (Cast, dtype, x) ->
      if real_or_complex dtype && real_or_complex (Nx.dtype x) then
        record t op x
      else nonlinear t op
  | Solve_triangular { a; b; _ } ->
      if owns t a then nonlinear t op else record t op b
  | Matmul (a, b) ->
      let sa = owns t a in
      if sa && owns t b then nonlinear t op
      else record t op (if sa then a else b)
  | Cat _ -> record_any t op
  | Gather _ -> record_any t op
  | Scatter _ -> record_any t op
  | Update _ -> record_any t op
  | Unfold _ -> record_any t op
  | Fold _ -> record_any t op
  | Fft _ -> record_any t op
  | Rfft _ -> record_any t op
  | Irfft _ -> record_any t op
  | Contiguous _ -> record_any t op
  | Move _ -> record_any t op
  | Place _ -> record_any t op
  | Read _ ->
      invalid_arg
        (Printf.sprintf
           "%s: a custom_jvp tangent map reads a tangent's value; under \
            reverse mode a tangent has none"
           t.entry)
  | Compare _ | Arg_reduce _ | Sort _ | Argsort _
  | Convert (Bitcast, _, _)
  | Threefry _ | Cholesky _ | Qr _ | Lu _ | Svd _ | Eig _ | Eigh _ ->
      nonlinear t op

let answer : type r. tape -> r Construct.t -> (unit -> r) option =
 fun t c ->
  match[@warning "@4@8"] c with
  | Detach x -> if owns t x then Some (fun () -> x) else None
  | Add (_, v) ->
      if owns t v then
        Some
          (fun () ->
            invalid_arg
              "Rune.Total.add: a custom_jvp tangent map adds a tangent under \
               reverse mode; a total takes values")
      else None
  | Lanes (_, x) ->
      if owns t x then
        Some
          (fun () ->
            invalid_arg
              (t.entry
             ^ ": reverse mode does not differentiate a value gathered across \
                a map's lanes"))
      else None
  | Scan _ | Remat _ | Barrier _ | Custom _ | Lane_index _ | Lane_count _ ->
      None

let install t f =
  Construct.install
    {
      op = Some { run = (fun op -> run t op); claims = (fun op -> claims t op) };
      call = (fun c -> answer t c);
    }
    f

(* Transposing *)

type cotangents = { tape : tape; cts : Nx.packed option array }

let cotangents t = { tape = t; cts = Array.make t.length None }

let add cts x ct =
  if owns cts.tape x then
    let i = index x in
    cts.cts.(i) <-
      Some
        (match cts.cts.(i) with
        | None -> Nx.P ct
        | Some prev -> Nx.P (Nx.add (Nx.unpack (Nx.dtype ct) prev) ct))

let cotangent cts x =
  if owns cts.tape x then Option.map (Nx.unpack (Nx.dtype x)) cts.cts.(index x)
  else None

(* [unbroadcast ct shape] is [ct] summed over the axes along which [shape] was
   broadcast to [ct]'s shape. *)
let unbroadcast ct shape =
  let ct_shape = Nx.shape ct in
  if ct_shape = shape then ct
  else
    let lead = Array.length ct_shape - Array.length shape in
    let axes =
      List.filter
        (fun i -> i < lead || (shape.(i - lead) = 1 && ct_shape.(i) <> 1))
        (List.init (Array.length ct_shape) Fun.id)
    in
    Nx.reshape shape (Nx.sum ~axes ct)

(* [shrink_axis ~axis (lo, hi) x] is [x] from [lo] to [hi] along [axis];
   [pad_axis ~axis (lo, hi) x] is [x] with [lo] and [hi] zeros around it along
   [axis]. *)
let shrink_axis ~axis (lo, hi) x =
  Nx.shrink
    (Array.mapi (fun i d -> if i = axis then (lo, hi) else (0, d)) (Nx.shape x))
    x

let pad_axis ~axis (lo, hi) x =
  Nx.pad
    (Array.mapi (fun i _ -> if i = axis then (lo, hi) else (0, 0)) (Nx.shape x))
    (Nx_dtype.zero (Nx.dtype x))
    x

(* The cotangent of the scattered updates: the cotangent at the positions they
   reach, less, under [`Set] with repeated indices, the updates a later one
   overwrites. *)
let scattered ~mode ~unique ~axis ~indices ~into ct =
  let g = Nx.take_along_axis ~axis ~indices ct in
  match mode with
  | `Set when not unique ->
      let shape = Nx.shape indices in
      let along =
        Array.mapi (fun i _ -> if i = axis then shape.(axis) else 1) shape
      in
      let rank =
        Nx.broadcast_to shape
          (Nx.reshape along (Nx.arange Nx.int32 0 shape.(axis) 1))
      in
      let winner =
        eval
          (Scatter
             {
               mode = `Set;
               unique = false;
               axis;
               indices;
               updates = rank;
               into = Nx.zeros Nx.int32 (Nx.shape into);
             })
      in
      Nx.where
        (Nx.equal (Nx.take_along_axis ~axis ~indices winner) rank)
        g (Nx.zeros_like g)
  | `Set | `Add -> g

(* The window of [ct] that [v], written at [starts], covers; each axis is read
   with a gather, so a traced [starts] stays traced. *)
let window ~starts v ct =
  let vshape = Nx.shape v in
  let rank = Array.length vshape in
  let w = ref ct in
  for axis = 0 to rank - 1 do
    let len = vshape.(axis) in
    let start = Nx.reshape [||] (Nx.slice [ Nx.I axis ] starts) in
    let idx = Nx.add (Nx.arange Nx.int32 0 len 1) start in
    let shape = Array.copy (Nx.shape !w) in
    shape.(axis) <- len;
    let along = Array.init rank (fun i -> if i = axis then len else 1) in
    w :=
      Nx.take_along_axis ~axis
        ~indices:(Nx.broadcast_to shape (Nx.reshape along idx))
        !w
  done;
  !w

let transpose_op : type a b.
    cotangents -> (a, b) Nx.t Nx.Op.t -> (a, b) Nx.t -> unit =
 fun cts op ct ->
  let owns x = owns cts.tape x and add x v = add cts x v in
  match[@warning "@4@8"] op with
  | Unary (_, x) -> add x (Nx.neg ct)
  | Binary (k, a, b) -> (
      match[@warning "@4@8"] (k : Nx_backend.binary) with
      | Add ->
          add a ct;
          add b ct
      | Sub ->
          add a ct;
          add b (Nx.neg ct)
      | Mul -> if owns a then add a (Nx.mul ct b) else add b (Nx.mul a ct)
      | Fdiv -> add a (Nx.div ct b)
      | Idiv | Mod | Pow | Atan2 | Maximum | Minimum | And | Or | Xor ->
          assert false (* Never recorded. *))
  | Where (c, a, b) ->
      let zeros = Nx.zeros_like ct in
      add a (Nx.where c ct zeros);
      add b (Nx.where c zeros ct)
  | Reduce (_, axes, x) ->
      let shape = Nx.shape x in
      let kept =
        Array.mapi (fun i d -> if Array.mem i axes then 1 else d) shape
      in
      add x (Nx.broadcast_to shape (Nx.reshape kept ct))
  | Scan (_, axis, x) ->
      let axes = [ (if axis < 0 then axis + Nx.ndim x else axis) ] in
      add x (Nx.flip ~axes (Nx.cumsum ~axis:(List.hd axes) (Nx.flip ~axes ct)))
  | Pad (padding, _, x) ->
      add x
        (Nx.shrink
           (Array.mapi (fun i (lo, _) -> (lo, lo + (Nx.shape x).(i))) padding)
           ct)
  | Cat (axis, xs) ->
      ignore
        (List.fold_left
           (fun lo x ->
             let hi = lo + (Nx.shape x).(axis) in
             add x (shrink_axis ~axis (lo, hi) ct);
             hi)
           0 xs)
  | Convert (_, _, x) -> add x (Nx.cast (Nx.dtype x) ct)
  | Gather (axis, indices, x) ->
      add x
        (eval
           (Scatter
              {
                mode = `Add;
                unique = false;
                axis;
                indices;
                updates = ct;
                into = Nx.zeros_like x;
              }))
  | Scatter { mode; unique; axis; indices; updates; into } ->
      add updates (scattered ~mode ~unique ~axis ~indices ~into ct);
      add into
        (match mode with
        | `Add -> ct
        | `Set ->
            Nx.mul ct
              (eval
                 (Scatter
                    {
                      mode = `Set;
                      unique;
                      axis;
                      indices;
                      updates = Nx.zeros_like updates;
                      into = Nx.ones_like into;
                    })))
  | Update (x, starts, v) ->
      add x (eval (Update (ct, starts, Nx.zeros_like v)));
      add v (window ~starts v ct)
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      let shape = Nx.shape x and k = Array.length kernel_size in
      let output_size = Array.sub shape (Array.length shape - k) k in
      add x
        (eval
           (Fold { output_size; kernel_size; stride; dilation; padding; x = ct }))
  | Fold { kernel_size; stride; dilation; padding; x; _ } ->
      add x (eval (Unfold { kernel_size; stride; dilation; padding; x = ct }))
  | Fft { inverse; axes; x } -> add x (eval (Fft { inverse; axes; x = ct }))
  | Rfft { axes; x; _ } ->
      (* rfft is a real embedding, an fft and a slice to the first n/2 + 1 bins
         of the last axis: each transposes to its adjoint, a zero pad, the fft
         itself and the real part. *)
      let last = axes.(Array.length axes - 1) in
      let n = (Nx.shape x).(last) and m = (Nx.shape ct).(last) in
      let ct = if n > m then pad_axis ~axis:last (0, n - m) ct else ct in
      add x
        (Nx.real (Nx.dtype x) (eval (Fft { inverse = false; axes; x = ct })))
  | Irfft { axes; x; _ } ->
      (* irfft extends the spectrum along the last axis by its conjugate mirror,
         runs the inverse fft and takes the real part. The transpose embeds the
         real cotangent, runs the same inverse fft and folds the mirror back:
         bins 1 .. n - m received a second contribution, which on a real
         cotangent's transform equals the first. *)
      let last = axes.(Array.length axes - 1) in
      let n = (Nx.shape ct).(last) in
      let m = (n / 2) + 1 in
      let z =
        eval (Fft { inverse = true; axes; x = Nx.cast (Nx.dtype x) ct })
      in
      let head = shrink_axis ~axis:last (0, m) z in
      add x
        (if n - m >= 1 then
           Nx.add head
             (pad_axis ~axis:last
                (1, m - 1 - (n - m))
                (shrink_axis ~axis:last (1, n - m + 1) head))
         else head)
  | Contiguous x -> add x ct
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      (* The solve with op(A) transposes to the solve with op(A)ᵀ, the conjugate
         of the other solve applied to the conjugate cotangent. *)
      add b
        (Nx.conjugate
           (eval
              (Solve_triangular
                 {
                   upper;
                   transpose = not transpose;
                   unit_diag;
                   a;
                   b = Nx.conjugate ct;
                 })))
  | Move (x, m) -> (
      match[@warning "@4@8"] m with
      | Reshape _ ->
          let ct = if Nx.is_c_contiguous ct then ct else Nx.contiguous ct in
          add x (Nx.reshape (Nx.shape x) ct)
      | Expand _ -> add x (unbroadcast ct (Nx.shape x))
      | Permute p ->
          let inverse = Array.make (Array.length p) 0 in
          Array.iteri (fun i j -> inverse.(j) <- i) p;
          add x (Nx.transpose ~axes:(Array.to_list inverse) ct)
      | Shrink limits ->
          let shape = Nx.shape x in
          add x
            (Nx.pad
               (Array.mapi (fun i (lo, hi) -> (lo, shape.(i) - hi)) limits)
               (Nx_dtype.zero (Nx.dtype x))
               ct)
      | Flip dims ->
          let axes =
            List.filter
              (fun i -> dims.(i))
              (List.init (Array.length dims) Fun.id)
          in
          add x (Nx.flip ~axes ct)
      | Window { axis; size; step } ->
          (* Overlap-add: input position [w * step + j] receives the cotangent
             of window [w] at offset [j], which is what fold sums. *)
          let shape = Nx.shape x in
          let r = Array.length shape in
          let to_fold =
            List.init (r + 1) (fun i ->
                if i < axis then i
                else if i <= r - 2 then i + 1
                else if i = r - 1 then r
                else axis)
          in
          let folded =
            eval
              (Fold
                 {
                   output_size = [| shape.(axis) |];
                   kernel_size = [| size |];
                   stride = [| step |];
                   dilation = [| 1 |];
                   padding = [| (0, 0) |];
                   x = Nx.transpose ~axes:to_fold ct;
                 })
          in
          let from_fold =
            List.init r (fun j ->
                if j < axis then j else if j = axis then r - 1 else j - 1)
          in
          add x (Nx.transpose ~axes:from_fold folded))
  | Matmul (a, b) ->
      if owns a then
        add a (unbroadcast (Nx.matmul ct (Nx.matrix_transpose b)) (Nx.shape a))
      else
        add b (unbroadcast (Nx.matmul (Nx.matrix_transpose a) ct) (Nx.shape b))
  | Place (_, x) -> add x (Nx.place (Nx.placement x) ct)
  | Compare _ | Sort _ | Threefry _ | Cholesky _ | Arg_reduce _ | Argsort _
  | Read _ ->
      assert false (* Never recorded. *)

let transpose_call cts i inputs like pullback =
  let received = ref false in
  let outputs =
    List.mapi
      (fun j (Nx.P y) ->
        match cts.cts.(i + j) with
        | Some ct ->
            received := true;
            ct
        | None -> Nx.P (Nx.zeros_like y))
      like
  in
  if !received then
    List.iter2
      (fun (Nx.P x) ct -> add cts x (Nx.unpack (Nx.dtype x) ct))
      inputs (pullback outputs)

let transpose cts =
  let t = cts.tape in
  for i = t.length - 1 downto 0 do
    match t.entries.(i) with
    | Recorded op ->
        Option.iter
          (fun ct -> transpose_op cts op (Nx.unpack (Nx.Op.dtype op) ct))
          cts.cts.(i)
    | Call { inputs; like; pullback } ->
        transpose_call cts i inputs like pullback
    | Input | Part -> ()
  done
