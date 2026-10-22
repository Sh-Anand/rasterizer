# Rasterizer notes

Record design decisions and their tradeoffs, not implementation details or progress.

## Renderer

Render on CUDA only. The CPU renderer, the replaceable stage interfaces and the CPU-hosted operation trace were removed: new features such as the ray-traced bounce and NTC decode are GPU-only, so a CPU copy would either diverge or need annotations for every shader feature, and a roughly right image checked by eye suffices for this research code. One renderer owns the stream, kernel module and every device buffer. Rasterization owns coverage and visibility; fragment shading consumes the resident fragments. Keep postprocessing on the CPU.

Prepare scene geometry independently of the camera. Decode attributes, generate missing flat normals and tangents (also regenerating a primitive's supplied tangents when any are invalid or a triangle mixes handedness, as in Sponza, instead of rejecting the scene), and split normal/tangent seams before vertex shading so it receives fixed topology and object-space attributes. Vertex shading applies the instance transforms and view projection without changing indices.

Process vertices once for the scene, then resolve visibility across all materials before shading. Retain one final fragment per render-resolution sample, replacing its attributes and material index when a closer fragment wins. This bounds fragment storage by the sample count and eliminates cross-material shading overdraw.

Shade the surviving fragments in one dispatch over framebuffer-order slots, skipping empty slots. Each fragment carries its material index; shading receives the full material table, with a missing index selecting the default glTF material.

Keep shading and sampling code as plain Rust shared by kernels and host preparation. All texture levels live in one texel array, and textures store offsets into it, so kernels shade from a flat array uploaded unchanged. Mip levels and the shadow map stay row-major: with 8x4 shading warps, 8x8 Z-ordered tiles removed only 3-5% more global-load sectors, and their index arithmetic cancelled that saving on the RTX 2000 Ada. Keep resource loading, mip generation and shadow fitting on the host. See [README.md](README.md#cuda) for CUDA setup and measurement scope.

Preserve equal-depth winners by processing the default material first, then ascending glTF material index, with stable primitive and triangle order. Strict less-than depth testing keeps the first winner. This traversal order does not create separate shading batches.

Each single-sample or SSAA fragment owns one distinct framebuffer sample, so shading writes its color without further depth arbitration. MSAA would require per-sample ownership masks. This design assumes no shader-side discard or depth replacement.

Shade fragments independently, without quad-based lane exchange. Use analytic perspective-correct UV gradients for texture sampling instead of neighboring fragment values. This avoids helper invocations for UV derivatives, but does not reproduce native quad-packing and helper-lane overhead; shaders needing derivatives of other computed values will need explicit derivative handling.

CUDA implements software rasterization on compute cores; it does not use fixed-function graphics rasterization. Treat fragment-to-lane packing as an explicit model, not an exact reproduction of a vendor's hardware scheduler. CUDA shading gives each warp an 8x4 pixel block rather than a 32x1 row strip: compared with strips it fetched 8-25% fewer global-load sectors and lost fewer lanes to divergence, while fragment attribute loads and color stores stay fully coalesced.

Measure stages with events on the same stream, without synchronizing between camera stages; report remaining frame wall time as setup/readback. Count vertices entering vertex shading and surviving fragments entering fragment shading; SSAA fragments are render-resolution samples. Report scene triangles before clipping, and keep loading, lighting preparation, framebuffer writeback, and output times separate from shader timings.

Sum measured backend setup, loading, geometry, materials, and shadow preparation as the one-time cost for a static scene. Sum vertex shading, rasterization, fragment shading, the bounce, render setup/writeback, and postprocessing as the per-render cost when the camera changes. Report PNG encoding and writing separately, outside both totals. These are grouped measurements of the current render, not a steady-state frame-loop benchmark.

Report post-clipping triangles but not vertices: deduplicating source vertices would cost a bitset clear and an atomic per surviving triangle vertex in every visibility pass, including shadow passes.

CUDA rasterization handles small triangle bounds within one thread. Larger triangles enter a bounded queue with at most one entry per input triangle; cooperative blocks divide their samples according to screen area. Warp reductions aggregate coverage/statistics updates. A fast path avoids clipping scratch for wholly visible triangles. This limits idle lanes and repeated setup for tiny triangles without tile bins or a global clipped-geometry pool.

Visibility atomically minimizes an ordered FP32 depth and stable triangle/fan rank packed into 64 bits. Signed zero compares equally and nonfinite depths are rejected. Triangles are stored in stable material order so a winning rank directly identifies its geometry; shadow passes use the same ranks because ties cannot change a depth-only result. One thread per framebuffer sample reconstructs and interpolates its winning face, initializing empty fragment slots in the same pass. This avoids a second rasterization of all triangles, but still repeats projection/clipping for visible samples. Shading writes black for empty samples.

The shadow pass uses the same visibility kernel, skips attribute emission, and unpacks winning depths directly into the resident shadow map. Camera output reads back RGB and depth only, plus small validation/statistics counters. Renders do not check for stale resources: prepare again after any scene or resource change.

## Workload traces

Take workload data from the GPU's own lane assignment rather than a CPU replay: kernels dump per-lane records, as the bounce writes each pixel's primary and hit materials, to be extended with decoder matrix operations. The removed CPU trace replayed the shared shader on CPU-rasterized fragments with source-level operation estimates. Deferred: per-lane addresses with coalescing and cache analysis.

## Clipping

Use FP32 for all rendering calculations, including clipping, coverage, interpolation, texture filtering, lighting, and image export.

We chose explicit clipping with synthetic ("fake") intersection vertices instead of a guard band.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.

## Shadows

Render shadow maps with the camera's vertex and visibility kernels. Depth passes use the existing shadow depth/bias rules, skip fragment collection and shading, and write directly into the map's depth buffer. Keep light-view construction and visibility filtering in the lighting module. Attribute transformation is shared with camera rendering; a position-only vertex path is deferred. Shadow work belongs to one-time preparation and leaves camera-pass statistics unchanged.

One orthographic shadow map covers the full scene so off-camera geometry can cast shadows. This keeps fitting simple, but large scenes spread the available resolution thinly. Use a one-texel slope bias to cover the bilinear PCF footprint, plus a configurable constant bias: too little bias causes self-shadowing artifacts; too much detaches shadows from their casters.

Use fixed 2x2 percentage-closer filtering: bilinearly blend four depth-comparison results, not depths. Apply fractional visibility to the light during shading. This reduces jagged edges but does not model a physical penumbra. Keep it independent of material-texture filtering; do not generate depth mipmaps. Clamp border taps.

## Lights

Light the scene with the configured directional light only. Emissive-triangle area lights were removed: their per-triangle cube shadow maps dominated preparation in scenes with emitters, and the ray-traced bounce already carries emitted light (counting both double-counted it). Emissive surfaces still glow, and light their surroundings only through bounce hits. The camera-aligned debug light was removed with them, leaving one light, so shading reads it from a kernel parameter rather than looping over a light list.

Render direct diffuse and specular lighting, shadows, and surface emission; the optional ray-traced bounce below is the only indirect lighting. Ignore material occlusion maps.

## Ray-traced bounce

Add one optional cosine-weighted diffuse bounce per pixel; rasterization keeps primary visibility, as in hybrid engines. The bounce exists to produce incoherent hit shading for tensor-core batching studies, so a roughly right image suffices.

A host BVH splits at the median centroid of the widest axis, with leaves of up to four triangles, and indexes the rank-ordered triangle list, so hits read the resident vertex output and primitive table without duplicated geometry. Surface-area-heuristic builds and child ordering are deferred. Hits interpolate vertex attributes with barycentrics, read texture level zero (no ray cones), and run the shared shader with the ray origin as viewer; misses add nothing.

The bounce kernel keeps shading's 8x4 pixel blocks, so warps shade hits in ray launch order, unsorted. A separate kernel leaves primary shading's code and timing unchanged. Each bounce render writes every pixel's primary and hit materials for batching analysis.

Rays per pixel are a build-time constant (`BOUNCE_RAYS`, default 4), so the kernel's ray loop has a fixed trip count; changing it rebuilds. Each lane traces its pixel's rays in sequence, so every loop iteration is one warp-wide batch of hits.

## Transparency

Treat every material as fully opaque. Ignore alpha mode, alpha cutoff, base-color alpha, and texture alpha in camera rendering and shadow maps. Rasterization performs no material texture sampling. Supporting cutouts or blending later requires revisiting visibility and depth ownership.

## Runtime material shading

Use one global material-texture filter: `none` (nearest texel of the nearest mip level), `bilinear` (level zero), or `trilinear` (mipmapped). `none` reads one texel per texture, as an on-sample neural decoder evaluates one texel; it is the initial NTC mode and its baseline, aliasing instead of the noise stochastic filtering would add. Override glTF filter choices but preserve wrap modes. Filter color textures in linear RGB; data maps stay linear. Alpha is ignored. Rasterized passes use analytic perspective-correct UV gradients instead of GPU quad derivatives; non-rasterized point queries use level zero.

Sample all of a material's textures together at one UV before lighting, as an on-sample neural texture decoder produces a whole texture set at once. A material's textures must share one texture-coordinate set; mixed sets are rejected. Fragments carry one UV and gradient pair instead of four, and the normal map is read even when an unusable tangent frame discards it.

Use isotropic box-filtered mipmaps, without anisotropic filtering or normal-variance roughness compensation. Oblique textures can blur. Shadow maps use comparison filtering independently of this setting.

Compute direct diffuse and specular together at runtime, sharing the light's direction and shadow lookup. Use GGX distribution, height-correlated Smith visibility, and Schlick Fresnel, with dielectric F0 = 0.04 and metallic F0 = base color. Multiply direct diffuse reflectance by `1 - F_dielectric(V dot H)`; do not apply the metallic-tinted specular Fresnel to this term. This follows the [glTF metallic-roughness mixing model](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metal-brdf-and-dielectric-brdf), itself an approximation rather than an exactly energy-conserving layered BRDF.

No environment reflections or microfacet multiple-scattering compensation is included. Rough metals can therefore lose energy and metals without directly visible lights can appear dark.

Clamp perceptual roughness to at least 0.045 for specular evaluation, avoiding a singular perfect-mirror lobe and reducing very small highlights. Zero-roughness materials are therefore approximated, not true mirrors.

Preserve the configured directional light's existing brightness convention: intensity 1 gives a facing white Lambertian reference a response of 1 before Fresnel weighting. Convert it to incident irradiance with a factor of pi so diffuse and specular use the same light units.

Prepare shadow maps once per process. Camera movement alone does not require new maps. Disk caching of shadows is deferred.

Keep the framebuffer in linear HDR and apply manual exposure, per-channel Reinhard tone mapping (`x / (1 + x)`), then sRGB encoding only at PNG export. This simple global curve preserves highlight differences but can desaturate bright colors; no automatic exposure or filmic color transform is modeled. Tone mapping can be disabled to recover clamped output.

Use regular-grid supersampling. A scale of N renders N times the output width and height, with independent coverage, depth, and shading for N² samples per pixel. Resolve color with a box average in linear HDR before tone mapping and sRGB encoding. This reuses the single-sample rasterizer but multiplies camera-framebuffer storage and pixel shading work by N²; it is not an MSAA execution model for GPU traces. Default to 1 (off). Shadow resolution is independent of supersampling.
