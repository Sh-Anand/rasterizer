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

Perspective shadow faces store linear light-view depth, normalized by their far distance, with a half-texel slope bias and the configured constant bias. Their range fits the full scene; the near plane is 0.0001 times the far distance, so very nearby occluders can be clipped. All light maps remain resident during direct-light baking.

Lighting and visibility will remain rasterization-based, including future GI; no ray tracing or shadow rays.

## Baked lighting

Bake direct diffuse lighting and exactly one diffuse indirect bounce for static geometry, materials, and lights. Store linear floating-point lighting for a white receiver; apply the receiver's base color and emission when rendering. This is not a bake of view-dependent specular lighting.

Generate a separate lightmap UV atlas with xatlas, giving each placed mesh instance its own space without changing material UVs or source triangle identities. Two-sided surfaces have independent front/back lighting. Chart padding supports bilinear filtering; sub-texel triangles use a centroid sample. The requested resolution is approximate: many small charts can make the atlas substantially larger.

For indirect lighting, rasterize a five-face hemicube at each surface sample and integrate incoming light with cosine/solid-angle weights. Captures read only the completed direct-light bake, multiplied by the visible surface's `base_color * (1 - metallic)`. This models only diffuse reflection: pure metals contribute no bounce, so specular indirect lighting is deliberately omitted. Exclude emission because emissive triangles already contribute to direct lighting; never feed indirect results back into these captures. Single-sided backfaces block light but do not reflect it.

Keep the initial bake coarse: this CPU method is expensive, and low lightmap/hemicube resolutions can blur shadows, miss small features, and leak light. Cache the result using scene, material, light, and bake settings; camera and image-resolution changes do not require rebaking.
