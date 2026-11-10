(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module K = Kernels
module Sub = Rig.Submission
module Run = Rig.Submission.Run

type 'p t = {
  kernel : int;
  size : int;
  base : int;
  refs : int array;
  gx : int;
  gy : int;
  gz : int;
  threads : int;
  shared : int;
  params : 'p;
}

type 'p call = {
  launches : 'p t array;
  buffers : Rig.Buffer.t array;
  workspace : int;
  tickets : int;
  kept : bool;
  frees : Rig.Buffer.t list;
}

type 'p plan = ('p call list, Nx_array.answer) result

(* Empty buffers, which allocate nothing, distinct from any other: the door
   binds the device's buffers in their place. *)
let workspace = Rig.Buffer.create Rig.host 0
let tickets = Rig.Buffer.create Rig.host 0

let kernel name =
  match Array.find_index (fun (n, _) -> n = name) K.kernels with
  | Some i -> i
  | None -> invalid_arg ("Launch.kernel: no kernel " ^ name)

let slot_refs ~header ~reads ~writes =
  let refs = Array.make (3 * (reads + writes)) 0 in
  for s = 0 to reads + writes - 1 do
    refs.(3 * s) <- 8 * (header + s);
    refs.((3 * s) + 1) <- s;
    if s >= reads then refs.((3 * s) + 2) <- 1
  done;
  refs

let bytes b r at =
  for j = 0 to (Bytes.length b / 8) - 1 do
    Run.int64 r at (8 * j) (Bytes.get_int64_le b (8 * j))
  done

let write w c r s =
  let ls = c.launches in
  for i = 0 to Array.length ls - 1 do
    let l = ls.(i) in
    let at = Sub.block s i in
    Run.groups r at l.gx l.gy l.gz;
    Run.threads r at l.threads 1 1;
    Run.shared r at l.shared;
    w l.params r at
  done

(* Keys: the count of launches, then each one's kernel, size, base, count of
   ref words and refs. *)

let key c =
  let ls = c.launches in
  let n = Array.fold_left (fun n l -> n + 4 + Array.length l.refs) 1 ls in
  let k = Array.make n (Array.length ls) in
  let at = ref 1 in
  Array.iter
    (fun l ->
      let i = !at and m = Array.length l.refs in
      k.(i) <- l.kernel;
      k.(i + 1) <- l.size;
      k.(i + 2) <- l.base;
      k.(i + 3) <- m;
      Array.blit l.refs 0 k (i + 4) m;
      at := i + 4 + m)
    ls;
  k

let rec same_refs k at refs j =
  j = Array.length refs || (k.(at + j) = refs.(j) && same_refs k at refs (j + 1))

(* Whether [ls] from launch [i] match [k] from [at]. *)
let rec same k at ls i =
  if i = Array.length ls then at = Array.length k
  else
    let l = ls.(i) in
    let m = Array.length l.refs in
    at + 4 + m <= Array.length k
    && k.(at) = l.kernel
    && k.(at + 1) = l.size
    && k.(at + 2) = l.base
    && k.(at + 3) = m
    && same_refs k (at + 4) l.refs 0
    && same k (at + 4 + m) ls (i + 1)

let matches k c =
  Array.length k > 0 && k.(0) = Array.length c.launches && same k 1 c.launches 0

let parts image c =
  Array.map
    (fun l ->
      let refs =
        Array.init
          (Array.length l.refs / 3)
          (fun r ->
            let at = l.refs.(3 * r) and slot = l.base + l.refs.((3 * r) + 1) in
            if l.refs.((3 * r) + 2) = 1 then Sub.write slot ~at
            else Sub.read slot ~at)
      in
      let name = fst K.kernels.(l.kernel) in
      match Rig.Image.resolve image name with
      | Some f -> Sub.launch ~refs f ~params:l.size
      | None -> invalid_arg ("Nx_cuda: the cubin has no " ^ name))
    c.launches
