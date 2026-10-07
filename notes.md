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

Bake exactly one diffuse indirect bounce for static geometry, materials, and lights. The saved lightmap contains no direct lighting. Store linear floating-point indirect lighting for a white receiver; apply the receiver's `base_color * (1 - metallic)` when rendering, alongside runtime direct lighting and emission.

Limit GI to one diffuse bounce to keep CPU baking tractable. Second and later reflections are omitted, so enclosed spaces can be too dark and multi-bounce color bleeding is missing. Increasing bake resolution does not recover these missing light paths.

Generate a separate lightmap UV atlas with xatlas, giving each placed mesh instance its own space without changing material UVs or source triangle identities. Two-sided surfaces have independent front/back lighting. Chart padding supports bilinear filtering; sub-texel triangles use a centroid sample. The requested resolution is approximate: many small charts can make the atlas substantially larger.

For indirect lighting, rasterize a five-face hemicube at each surface sample and integrate incoming light with cosine/solid-angle weights. Captures read a temporary direct Lambertian lighting atlas, multiplied by the visible surface's `base_color * (1 - metallic)`. Discard the direct atlas after computing the bounce; do not add it to the saved map. This models only diffuse reflection: pure metals contribute no bounce, so specular indirect lighting is deliberately omitted. Exclude emission because emissive triangles already contribute to runtime direct lighting; never feed indirect results back into these captures. Single-sided backfaces block light but do not reflect it.

Keep the initial bake coarse: this CPU method is expensive, and low lightmap/hemicube resolutions can blur shadows, miss small features, and leak light. Cache the result using scene, material, light, and bake settings; camera and image-resolution changes do not require rebaking.

The indirect bake remains Lambertian: both bounce captures and the final receiver omit angle-dependent diffuse Fresnel weighting. Its RGB lightmap does not retain incoming light directions, so removing this approximation would require storing directional lighting and doing more runtime work. The bake uses mesh normals; indirect normal-map detail would need higher-resolution normal-aware baking or directional lightmaps. Normal maps remain deferred.

## Runtime material shading

Compute direct diffuse and specular together at runtime, sharing each light's direction and shadow lookup. This adds per-pixel diffuse evaluation instead of baking it, but permits camera/light-dependent Fresnel weighting. Use GGX distribution, height-correlated Smith visibility, and Schlick Fresnel, with dielectric F0 = 0.04 and metallic F0 = base color. Multiply direct diffuse reflectance by `1 - F_dielectric(V dot H)`; do not apply the metallic-tinted specular Fresnel to this term. Share the underlying `base_color * (1 - metallic)` reflectance with indirect shading. This follows the [glTF metallic-roughness mixing model](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metal-brdf-and-dielectric-brdf), itself an approximation rather than an exactly energy-conserving layered BRDF.

No specular indirect lighting, environment reflections, or microfacet multiple-scattering compensation is included. Rough metals can therefore lose energy and metals without directly visible lights can appear dark.

Clamp perceptual roughness to at least 0.045 for specular evaluation, avoiding a singular perfect-mirror lobe and reducing very small highlights. Zero-roughness materials are therefore approximated, not true mirrors.

Preserve the configured directional light's existing brightness convention: intensity 1 gives a facing white Lambertian reference a response of 1 before Fresnel weighting. Convert it to incident irradiance with a factor of pi so diffuse and specular use the same light units.

Prepare shadow maps once per process and share them between baking and runtime direct lighting. They remain resident through rendering, increasing memory use during the indirect bake. Cached-lightmap renders still rebuild these maps at startup; camera movement alone does not require new maps or a new indirect bake. Disk caching of shadows is deferred.

Keep the existing clamped sRGB PNG output for now. Bright HDR highlights can clip; exposure and tone mapping are separate work.
