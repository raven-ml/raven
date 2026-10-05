(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let setting = Setting.value

let ring_allreduce_threshold =
  Setting.int ~reach:Output "RING_ALLREDUCE_THRESHOLD" 256_000

(* The lists combined below hold one value per device or per chunk. *)
let nonempty f = function
  | x :: rest -> f x rest
  | [] -> invalid_arg "an allreduce across no devices"

let handle_allreduce red =
  let buf = nth red 0 in
  match (device buf, arg red) with
  | Some (Multi devices), Allreduce { op; device } ->
      let d = Array.of_list devices in
      let ndev = Array.length d and shape = Ops.shape buf in
      let numel = Sint.prod shape in
      let fold = nonempty (List.fold_left (fun x y -> alu x op [ y ])) in
      let to_device ?shard i x = copy_to_device ?shard x (Single d.(i)) in
      let range (s, e) = [ Some (Int s, Int e) ] in
      let reassemble chunks copied =
        let padded =
          List.map2
            (fun (s, e) c -> pad c [ Some (Int s, Sint.(numel - Int e)) ])
            chunks copied
        in
        reshape (nonempty usum padded) shape
      in
      (* A ring allreduce gains nothing over the naive one with two devices or
         below 256k elements, and costs dispatches, chunking and reassembly. *)
      let concrete =
        List.for_all (function Int _ -> true | Sym _ -> false) shape
      in
      let large () =
        ndev > 2
        && Sint.(resolve (numel > Int (setting ring_allreduce_threshold)))
      in
      let use_all2all =
        concrete
        && (setting Setting.all2all >= 2
           || (large () && setting Setting.all2all >= 1))
      in
      let use_ring =
        concrete && (not use_all2all)
        && (setting Setting.ring >= 2 || (large () && setting Setting.ring >= 1))
      in
      if setting Setting.debug >= 2 then
        Format.printf "%s ALLREDUCE %dx%a | %a@."
          (if use_all2all then "ALL2ALL"
           else if use_ring then "RING"
           else "NAIVE")
          ndev Sint.pp numel Dtype.pp (dtype buf);
      let buf = pad_to buf (List.map (fun n -> Some (Int n)) (max_shape buf)) in
      (* Contiguous before it is copied. *)
      let buf = contiguous buf in
      let hdev = setting Setting.allreduce_node_ndevs in
      Some
        (match numel with
        | Int numel when concrete && hdev > 0 && ndev mod hdev = 0 ->
            let flat = reshape buf [ Int numel ] in
            let boxes =
              List.init (ndev / hdev) (fun b ->
                  List.init hdev (fun k -> (b * hdev) + k))
            in
            let cs =
              List.init hdev (fun k ->
                  (numel * k / hdev, numel * (k + 1) / hdev))
            in
            let owned = Array.make ndev flat
            and summed = Array.make ndev flat in
            List.iter
              (fun box ->
                List.iteri
                  (fun k i ->
                    let chunk j =
                      shrink (mselect flat j) (range (List.nth cs k))
                    in
                    owned.(i) <-
                      fold (List.map (fun j -> to_device i (chunk j)) box))
                  box)
              boxes;
            for k = 0 to hdev - 1 do
              let rank = List.map (fun box -> List.nth box k) boxes in
              List.iter
                (fun i ->
                  summed.(i) <-
                    fold
                      (owned.(i)
                      :: List.filter_map
                           (fun j ->
                             if j = i then None
                             else Some (to_device i owned.(j)))
                           rank))
                rank
            done;
            (* Device [k] of the first node holds chunk [k] reduced. *)
            let gathered =
              List.init hdev (fun k ->
                  match device with
                  | Single _ -> copy_to_device summed.(k) device
                  | Multi _ ->
                      nonempty mstack
                        (List.concat_map
                           (fun box ->
                             List.map
                               (fun j -> to_device j summed.(List.nth box k))
                               box)
                           boxes))
            in
            reassemble cs gathered
        | Int numel when use_ring || use_all2all ->
            (* Chunks of whole multiples of a small power of two. *)
            let factor =
              Option.value ~default:1
                (List.find_opt (fun f -> numel mod f = 0) [ 32; 16; 8; 4; 2 ])
            in
            let base = numel / factor / ndev
            and left = numel / factor mod ndev in
            let _, rev_chunks =
              List.fold_left
                (fun (s, acc) i ->
                  let e =
                    s + ((if i < left then base + 1 else base) * factor)
                  in
                  (e, (s, e) :: acc))
                (0, []) (List.init ndev Fun.id)
            in
            let chunks = List.rev rev_chunks in
            let flat = reshape buf [ Int numel ] in
            (* Reduce-scatter. *)
            let reduced_chunks =
              List.mapi
                (fun i c ->
                  if use_all2all then
                    fold
                      (List.init ndev (fun j ->
                           to_device i
                             (shrink
                                (reshape (mselect buf j) [ Int numel ])
                                (range c))))
                  else
                    let chunk = shrink flat (range c) in
                    let reduced = ref chunk in
                    for step = 0 to ndev - 2 do
                      let src = (i + step) mod ndev
                      and dest = (i + step + 1) mod ndev in
                      let shard =
                        match Ops.device !reduced with
                        | Some (Multi _) -> Some src
                        | _ -> None
                      in
                      reduced :=
                        alu
                          (to_device ?shard dest !reduced)
                          op
                          [ to_device ~shard:dest dest chunk ]
                    done;
                    !reduced)
                chunks
            in
            (* Allgather. *)
            let copied_chunks =
              List.mapi
                (fun i rc ->
                  match device with
                  | Single _ -> copy_to_device rc device
                  | Multi _ when use_all2all ->
                      nonempty mstack (List.init ndev (fun j -> to_device j rc))
                  | Multi _ ->
                      let chain = Array.make ndev rc in
                      for step = 0 to ndev - 2 do
                        chain.(step + 1) <-
                          to_device ((i + step) mod ndev) chain.(step)
                      done;
                      nonempty mstack
                        (List.init ndev (fun j ->
                             chain.(Helpers.floormod (j - i + 1) ndev))))
                reduced_chunks
            in
            reassemble chunks copied_chunks
        | _ ->
            (* Naive: copy to every device; a later shrink is handled there. *)
            shrink_to
              (fold
                 (List.init ndev (fun i ->
                      copy_to_device (mselect buf i) device)))
              (List.map Option.some shape))
  | _ -> None

let create_allreduce_function red =
  match arg red with
  | Allreduce { op; device } ->
      let output =
        v Op.Alloc
          ~src:(device_range_src (Some device))
          ~arg:
            (Param
               (param_arg ~slot:(unique_num ()) ~size:(max_numel red) ~device
                  (dtype red)))
      in
      let output =
        shrink_to
          (reshape output (List.map (fun n -> Int n) (max_shape red)))
          (List.map Option.some (shape red))
      in
      let buf = nth red 0 in
      let dst = param_like red 0 and src = param_like buf 1 in
      (* The allreduce of a parameter on several devices always has a value. *)
      let value = Option.get (handle_allreduce (allreduce src op device)) in
      let body = sink [ after dst [ store dst value ] ] in
      after output
        [
          call ~name:"allreduce" ~precompile:true body
            [ base output; contiguous buf ];
        ]
  | _ -> invalid_arg "not an allreduce"
