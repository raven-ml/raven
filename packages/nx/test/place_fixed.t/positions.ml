open Walk

let params = { w = Nx.ones Nx.float32 [| 2 |]; b = None }
let placed = Nx.Ptree.place_fixed walk_linear Nx.Host.on params
