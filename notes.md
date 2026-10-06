# Rasterizer notes

Record design decisions and their tradeoffs, not implementation details or progress.

## Clipping

We chose explicit clipping with synthetic ("fake") intersection vertices instead of a guard band.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.

## Shadows

One orthographic shadow map covers the full scene so off-camera geometry can cast shadows. This keeps fitting simple, but large scenes spread the available resolution thinly. We use hard shadows with a half-texel slope bias plus a configurable constant bias: too little bias causes self-shadowing artifacts; too much detaches shadows from their casters.

## Area lights

Use one light list for the configured directional light and emissive triangles from the full, unclipped scene. Emissive triangles remain visible geometry as well as light sources.

Approximate each triangle's diffuse illumination with one sample at its world-space centroid, weighted by its area, inverse squared distance, and the receiver and emitter cosines. Emission textures and alpha masks are also sampled at the centroid; sidedness follows the material. This is cheap but inaccurate near large triangles and depends on tessellation. More samples or area integration can replace it later.

Use rasterized shadow maps at each emitting triangle's centroid: five cube faces aligned to its normal for a one-sided emitter, six for a two-sided emitter. Each sample produces a hard shadow; this does not integrate visibility over the emitting area or model indirect lighting. Rendering and storage scale with the number of emitting triangles. We keep the per-triangle baseline rather than grouping emitters for now.

Perspective shadow faces store linear light-view depth, normalized by their far distance, with a half-texel slope bias and the configured constant bias. Their range fits the full scene; the near plane is 0.0001 times the far distance, so very nearby occluders can be clipped. The first implementation retains all light maps for the shading pass.

Lighting and visibility will remain rasterization-based, including future GI; no ray tracing or shadow rays.
