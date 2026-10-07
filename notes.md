# Rasterizer notes

Record design decisions and their tradeoffs, not implementation details or progress.

## Clipping

We chose explicit clipping with synthetic ("fake") intersection vertices instead of a guard band.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.

## Shadows

One orthographic shadow map covers the full scene so off-camera geometry can cast shadows. This keeps fitting simple, but large scenes spread the available resolution thinly. We use hard shadows with a half-texel slope bias plus a configurable constant bias: too little bias causes self-shadowing artifacts; too much detaches shadows from their casters.

## Area lights

Use one light list for the configured directional light and emissive triangles from the full, unclipped scene. Emissive triangles remain visible geometry as well as light sources.

Approximate each triangle's illumination with one sample at its world-space centroid, weighted by its area, inverse squared distance, and the receiver and emitter cosines. Emission textures and alpha masks are also sampled at the centroid; sidedness follows the material. This is cheap but inaccurate near large triangles and depends on tessellation. Specular highlights also use the centroid, so they do not reproduce the emitter's shape and can be especially inaccurate on smooth surfaces. More samples or area integration can replace it later.

Use rasterized shadow maps at each emitting triangle's centroid: five cube faces aligned to its normal for a one-sided emitter, six for a two-sided emitter. Each sample produces a hard shadow; this does not integrate visibility over the emitting area or model indirect lighting. Rendering and storage scale with the number of emitting triangles. We keep the per-triangle baseline rather than grouping emitters for now.

Perspective shadow faces store linear light-view depth, normalized by their far distance, with a half-texel slope bias and the configured constant bias. Their range fits the full scene; the near plane is 0.0001 times the far distance, so very nearby occluders can be clipped.

Lighting and visibility will remain rasterization-based, including future GI; no ray tracing or shadow rays.

## Baked lighting

Bake direct diffuse lighting and exactly one diffuse indirect bounce for static geometry, materials, and lights. Store linear floating-point lighting for a white receiver; apply the receiver's `base_color * (1 - metallic)` and add emission when rendering. This is not a bake of view-dependent specular lighting.

Limit GI to one diffuse bounce to keep CPU baking tractable. Second and later reflections are omitted, so enclosed spaces can be too dark and multi-bounce color bleeding is missing. Increasing bake resolution does not recover these missing light paths.

Generate a separate lightmap UV atlas with xatlas, giving each placed mesh instance its own space without changing material UVs or source triangle identities. Two-sided surfaces have independent front/back lighting. Chart padding supports bilinear filtering; sub-texel triangles use a centroid sample. The requested resolution is approximate: many small charts can make the atlas substantially larger.

For indirect lighting, rasterize a five-face hemicube at each surface sample and integrate incoming light with cosine/solid-angle weights. Captures read only the completed direct-light bake, multiplied by the visible surface's `base_color * (1 - metallic)`. This models only diffuse reflection: pure metals contribute no bounce, so specular indirect lighting is deliberately omitted. Exclude emission because emissive triangles already contribute to direct lighting; never feed indirect results back into these captures. Single-sided backfaces block light but do not reflect it.

Keep the initial bake coarse: this CPU method is expensive, and low lightmap/hemicube resolutions can blur shadows, miss small features, and leak light. Cache the result using scene, material, light, and bake settings; camera and image-resolution changes do not require rebaking.

The RGB lightmap combines direct and indirect lighting evaluated for the mesh normals; it does not retain incoming light directions. Per-pixel AO would require separating the two contributions. Normal-map detail must either be evaluated during baking (limited by atlas resolution), or use directional lightmaps/runtime diffuse lighting. These are deferred, not silently approximated by multiplying the combined map.

## Runtime material shading

Use baked Lambertian diffuse plus runtime direct specular: GGX distribution, height-correlated Smith visibility, and Schlick Fresnel, with dielectric F0 = 0.04 and metallic F0 = base color. Share diffuse reflectance between the bake and final shading. Keep the diffuse term independent of Fresnel because the bake lacks individual light directions; this is a practical approximation, not an exactly energy-conserving coupled BRDF. No specular indirect lighting, environment reflections, or microfacet multiple-scattering compensation is included. Rough metals can therefore lose energy and metals without directly visible lights can appear dark. The BRDF baseline follows [Filament's standard model](https://google.github.io/filament/main/filament.html#materialsystem/standardmodelsummary).

Clamp perceptual roughness to at least 0.045 for specular evaluation, avoiding a singular perfect-mirror lobe and reducing very small highlights. Zero-roughness materials are therefore approximated, not true mirrors.

Preserve the configured directional light's existing brightness convention: intensity 1 gives a facing white Lambertian surface a response of 1. Convert it to incident irradiance with a factor of pi so diffuse and specular use the same light units.

Prepare shadow maps once per process and share them between baking and runtime specular. They remain resident through rendering, increasing memory use during the indirect bake. Cached-lightmap renders still rebuild these maps at startup; camera movement alone does not require new maps or a new diffuse bake. Disk caching of shadows is deferred.

Keep the existing clamped sRGB PNG output for now. Bright HDR highlights can clip; exposure and tone mapping are separate work.
