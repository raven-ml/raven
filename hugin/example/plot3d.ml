open Ndarray

let create_helix_data () =
  let t = Ndarray.linspace float32 0. (4. *. Float.pi) 100 in
  let x = Ndarray.map (fun t -> Float.cos t) t in
  let y = Ndarray.map (fun t -> Float.sin t) t in
  let z = Ndarray.map (fun t -> t /. (4. *. Float.pi)) t in
  (x, y, z)

let () =
  let x, y, z = create_helix_data () in
  let fig =
    Hugin.plot3d ~title:"3D Helix" ~xlabel:"x" ~ylabel:"y" ~zlabel:"z" x y z
  in
  Hugin.show fig
