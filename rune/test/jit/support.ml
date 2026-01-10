(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

[@@@ocaml.warning "-32"]

open Rune_jit

let fresh_var_reset () = Ir.Var.counter := 0

let simple_add_graph () =
  fresh_var_reset ();
  let vA = Ir.Var.fresh () and vB = Ir.Var.fresh () and vC = Ir.Var.fresh () in
  let shape_int = [| 4 |] in
  let shape = Shape_expr.of_int_array shape_int in
  let meta dtype =
    {
      Ir.dtype = Ir.Dtype.Any_Dtype dtype;
      shape = shape_int;
      shape_expr = None;
      device = None;
    }
  in
  let nodes =
    [
      Ir.Any_Node (Placeholder { out_var = vA; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = vB; dtype = Float32; shape });
      Ir.Any_Node
        (Binop
           { op = Add; a_var = vA; b_var = vB; out_var = vC; dtype = Float32 });
    ]
  in
  let vars_md = Hashtbl.create 3 in
  List.iter (fun v -> Hashtbl.add vars_md v (meta Float32)) [ vA; vB; vC ];
  ( vA,
    vB,
    vC,
    {
      Ir.nodes;
      vars_metadata = vars_md;
      input_vars = [ vA; vB ];
      output_vars = [ vC ];
      symbolic_vars = [];
    } )

let simple_mulacc_graph () =
  fresh_var_reset ();
  let vA = Ir.Var.fresh ()
  and vB = Ir.Var.fresh ()
  and vC = Ir.Var.fresh ()
  and vOut = Ir.Var.fresh () in
  let shape_int = [| 4 |] in
  let shape = Shape_expr.of_int_array shape_int in
  let meta dtype =
    {
      Ir.dtype = Ir.Dtype.Any_Dtype dtype;
      shape = shape_int;
      shape_expr = None;
      device = None;
    }
  in
  let nodes =
    [
      Ir.Any_Node (Placeholder { out_var = vA; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = vB; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = vC; dtype = Float32; shape });
      Ir.Any_Node
        (Ternary
           {
             op = Mulacc;
             a_var = vA;
             b_var = vB;
             c_var = vC;
             out_var = vOut;
             dtype = Float32;
           });
    ]
  in
  let vars_md = Hashtbl.create 4 in
  List.iter (fun v -> Hashtbl.add vars_md v (meta Float32)) [ vA; vB; vC; vOut ];
  ( vA,
    vB,
    vC,
    vOut,
    {
      Ir.nodes;
      vars_metadata = vars_md;
      input_vars = [ vA; vB; vC ];
      output_vars = [ vOut ];
      symbolic_vars = [];
    } )

let simple_where_graph () =
  fresh_var_reset ();
  let vCond = Ir.Var.fresh ()
  and vX = Ir.Var.fresh ()
  and vY = Ir.Var.fresh ()
  and vOut = Ir.Var.fresh () in
  let shape_int = [| 4 |] in
  let shape = Shape_expr.of_int_array shape_int in
  let meta dtype =
    {
      Ir.dtype = Ir.Dtype.Any_Dtype dtype;
      shape = shape_int;
      shape_expr = None;
      device = None;
    }
  in
  let nodes =
    [
      Ir.Any_Node (Placeholder { out_var = vCond; dtype = Bool; shape });
      Ir.Any_Node (Placeholder { out_var = vX; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = vY; dtype = Float32; shape });
      Ir.Any_Node
        (Ternary
           {
             op = Where;
             a_var = vCond;
             b_var = vX;
             c_var = vY;
             out_var = vOut;
             dtype = Float32;
           });
    ]
  in
  let vars_md = Hashtbl.create 4 in
  Hashtbl.add vars_md vCond (meta Bool);
  List.iter (fun v -> Hashtbl.add vars_md v (meta Float32)) [ vX; vY; vOut ];
  ( vCond,
    vX,
    vY,
    vOut,
    {
      Ir.nodes;
      vars_metadata = vars_md;
      input_vars = [ vCond; vX; vY ];
      output_vars = [ vOut ];
      symbolic_vars = [];
    } )

let simple_mulacc_graph () =
  fresh_var_reset ();
  let vA = Ir.Var.fresh ()
  and vB = Ir.Var.fresh ()
  and vC = Ir.Var.fresh ()
  and vOut = Ir.Var.fresh () in
  let shape_int = [| 4 |] in
  let shape = Shape_expr.of_int_array shape_int in
  let meta dtype =
    {
      Ir.dtype = Ir.Dtype.Any_Dtype dtype;
      shape = shape_int;
      shape_expr = None;
      device = None;
    }
  in
  let nodes =
    [
      Ir.Any_Node (Placeholder { out_var = vA; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = vB; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = vC; dtype = Float32; shape });
      Ir.Any_Node
        (Ternary
           {
             op = Mulacc;
             a_var = vA;
             b_var = vB;
             c_var = vC;
             out_var = vOut;
             dtype = Float32;
           });
    ]
  in
  let vars_md = Hashtbl.create 4 in
  List.iter (fun v -> Hashtbl.add vars_md v (meta Float32)) [ vA; vB; vC; vOut ];
  ( vA,
    vB,
    vC,
    vOut,
    {
      Ir.nodes;
      vars_metadata = vars_md;
      input_vars = [ vA; vB; vC ];
      output_vars = [ vOut ];
      symbolic_vars = [];
    } )
