// A vertex function: Metal compiles and loads it, but a compute pipeline
// runs only kernels, so no pipeline of it can be made.

vertex float4 position(uint i [[vertex_id]]) {
  return float4(float(i), 0.0, 0.0, 1.0);
}
