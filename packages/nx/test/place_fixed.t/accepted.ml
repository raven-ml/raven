open Walk

let steps = { count = Nx.create Nx.int32 [| 1 |] [| 7l |] }
let on p x = Nx.Ptree.place_fixed walk_steps p x
let placed = on Nx.Host.on steps
