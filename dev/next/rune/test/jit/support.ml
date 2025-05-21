open Rune_jit

let fresh_var_reset () = Ir.Var.counter := 0

let simple_add_graph () =
  fresh_var_reset ();
  let vA = Ir.Var.fresh () and vB = Ir.Var.fresh () and vC = Ir.Var.fresh () in
  let shape = [| 4 |] in
  let meta dtype = { Ir.dtype = Ir.Dtype.Any_Dtype dtype; shape } in
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
    } )
