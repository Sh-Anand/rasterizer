# Rasterizer notes

Record design decisions and their tradeoffs, not implementation details or progress.

## GPU backend

Select the CPU or CUDA backend for all three stages together. Mixed CPU/CUDA stage combinations were removed: each paid host packing and transfers, and their extra code served no comparison. Keep postprocessing on the CPU. GPU fragment work consumes generated fragments; rasterization owns coverage and visibility. Keep the CPU renderer as the correctness reference.

Prepare scene geometry independently of the camera. Decode attributes, generate missing flat normals and tangents (also regenerating a primitive's supplied tangents when any are invalid or a triangle mixes handedness, as in Sponza, instead of rejecting the scene), and split normal/tangent seams before vertex shading so every backend receives fixed topology and object-space attributes. Vertex shading applies the instance transforms and view projection without changing indices. Keep the existing matrix multiplication and normalization order for the CPU reference.

The replaceable stage interfaces exchange host-resident geometry, fragment attributes, and linear HDR colors; only CPU stages implement them. The CUDA specialization of the same Pipeline keeps these intermediates on one device and stream. Rasterization owns clipping, projection, interpolation, and depth; fragment shading owns material color evaluation. The coordinator allocates the framebuffer and scatters returned colors. Supersampling resolve, exposure, tone mapping, and export remain ordinary framebuffer operations outside the replaceable stages.

Process vertices once for the scene, then resolve visibility across all materials before shading. Retain one final fragment per render-resolution sample, replacing its attributes and material index when a closer fragment wins. This bounds fragment storage by the sample count and eliminates cross-material shading overdraw.

Invoke the fragment stage once for the surviving fragments. The CPU collects compact winners; CUDA uses framebuffer-order slots and skips empty slots in its shading kernel. Each fragment carries its material index; shading receives the full material table, with a missing index selecting the default glTF material. The GPU backend uses one dispatch with access to all referenced material and texture data.

Share shading and sampling algorithms and resource layouts between CPU and CUDA. All texture levels live in one texel array and all shadow maps in one depth atlas; textures and maps store offsets, so both backends shade from the same flat arrays and CUDA uploads them unchanged. Mip levels and shadow maps stay row-major: with 8x4 shading warps, 8x8 Z-ordered tiles removed only 3-5% more global-load sectors, and their index arithmetic cancelled that saving on the RTX 2000 Ada. Keep resource loading, mip generation, light discovery and shadow fitting on the host; render shadow depths on the selected backend. See [README.md](README.md#cuda-backends) for CUDA setup, resource invalidation, and measurement scope.

Preserve equal-depth winners by processing the default material first, then ascending glTF material index, with stable primitive and triangle order. Strict less-than depth testing keeps the first winner. This traversal order does not create separate shading batches; the CPU reference uses the same order for comparisons.

Each single-sample or SSAA fragment owns one distinct framebuffer sample. The coordinator writes its returned color without further depth arbitration. MSAA would require per-sample ownership masks. This design assumes no shader-side discard or depth replacement.

Shade fragments independently, without quad-based lane exchange. Use analytic perspective-correct UV gradients for texture sampling instead of neighboring fragment values. This avoids helper invocations for UV derivatives, but does not reproduce native quad-packing and helper-lane overhead; shaders needing derivatives of other computed values will need explicit derivative handling.

CUDA implements software rasterization on compute cores; it does not use fixed-function graphics rasterization. Treat fragment-to-lane packing as an explicit model, not an exact reproduction of a vendor's hardware scheduler. CUDA shading gives each warp an 8x4 pixel block rather than a 32x1 row strip: compared with strips it fetched 8-25% fewer global-load sectors and lost fewer lanes to divergence, while fragment attribute loads and color stores stay fully coalesced.

Measure wall time around CPU stage interfaces. Measure CUDA stages with events on the same stream, without synchronizing between camera stages; report remaining frame wall time as setup/readback. Count vertices entering vertex shading and surviving fragments entering fragment shading; SSAA fragments are render-resolution samples. Report scene triangles before clipping, and keep loading, lighting preparation, framebuffer writeback, and output times separate from shader timings.

Sum measured backend setup, loading, geometry, materials, and shadow preparation as the one-time cost for a static scene. Sum vertex shading, rasterization, fragment shading, render setup/writeback, and postprocessing as the per-render cost when the camera changes. Report PNG encoding and writing separately, outside both totals. These are grouped measurements of the current render, not a steady-state frame-loop benchmark.

All CPU stages share a Rayon worker pool using all available logical CPUs. Vertex and fragment shading preserve output order. Rasterization projects primitives in parallel, bins triangles by their covered sample rows, then processes disjoint bands in parallel. Each band retains triangle order and owns its depth and fragment storage; no pixel locks are needed. The shadow-map loop remains sequential, with parallel vertex shading and rasterization inside each pass.

The CPU projects each surviving source vertex once and shares its index across triangles. Clipping appends intersection vertices while retaining source indices for unchanged vertices. The reported post-clipping vertex count measures unique surviving source vertices plus generated intersections, not triangle corners. CUDA does not report this count: deduplicating source vertices cost a bitset clear and an atomic per surviving triangle vertex in every visibility pass, including shadow passes. CUDA still reports post-clipping triangles.

CUDA rasterization handles small triangle bounds within one thread. Larger triangles enter a bounded queue with at most one entry per input triangle; cooperative blocks divide their samples according to screen area. Warp reductions aggregate coverage/statistics updates. A fast path avoids clipping scratch for wholly visible triangles. This limits idle lanes and repeated setup for tiny triangles without tile bins or a global clipped-geometry pool.

Visibility atomically minimizes an ordered FP32 depth and stable triangle/fan rank packed into 64 bits. Signed zero compares equally and nonfinite depths are rejected. Triangles are stored in stable material order so a winning rank directly identifies its geometry; shadow passes use the same ranks because ties cannot change a depth-only result. One thread per framebuffer sample reconstructs and interpolates its winning face, initializing empty fragment slots in the same pass. This avoids a second rasterization of all triangles, but still repeats projection/clipping for visible samples. Shading writes black for empty samples.

Shadow passes use the same visibility kernel, skip attribute emission, and unpack winning depths directly into the resident shadow atlas. Camera output reads back RGB and depth only, plus small validation/statistics counters. CUDA stage instances must share one device; invalidation requires preparing resident lights/shadows again before rendering.

## Trace backend

Emit a binary operation trace of fragment shading for workload analysis; no image is required. The trace follows the shared GPU shader, evaluating values only to resolve branches. Lanes use the CUDA packing: 32-lane warps covering 8x4 pixel blocks. Emit one record per operation and active mask: diverging lanes produce separate records, while lanes running the same code on different data share one.

Texture texels, shadow depths, descriptors and fragment inputs are all plain loads; each descriptor read counts as one load. Straight-line arithmetic is grouped into FP32, special-function and integer counts. Counts are source-level estimates of shader work, not compiled instructions; compiler artifacts such as local-memory copies are excluded.

The trace runs the CPU vertex and rasterization stages and hooks regions inside the shared shader, so only branches between regions split masks: texture presence, the second trilinear level, facing lights, shadowed lights and visible lights. Branches inside a region (texture wrap modes, normal-map early exits, extra shadow taps at cube-face edges and corners, BRDF early returns) are folded into that region's counts. CUDA shading passes a no-op recorder; its generated code and timing are unchanged.

Deferred: per-lane addresses with coalescing and cache analysis, which buffer each load reads, and per-lane texture or weight identity for decoder batching.

## Clipping

Use FP32 for all rendering calculations, including clipping, coverage, interpolation, texture filtering, lighting, and image export.

We chose explicit clipping with synthetic ("fake") intersection vertices instead of a guard band.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.

## Shadows

Generate shadow maps through the camera pipeline's selected vertex and rasterization backends. Depth passes use the existing shadow depth/bias rules, skip fragment collection and shading, and write directly into the map's depth buffer. Keep light-view construction and visibility filtering in the shadow module. Attribute transformation is shared with camera rendering; a position-only vertex path is deferred. Shadow work belongs to one-time preparation and leaves camera-pass statistics unchanged.

One orthographic shadow map covers the full scene so off-camera geometry can cast shadows. This keeps fitting simple, but large scenes spread the available resolution thinly. Use a one-texel slope bias to cover the bilinear PCF footprint, plus a configurable constant bias: too little bias causes self-shadowing artifacts; too much detaches shadows from their casters.

Use fixed 2x2 percentage-closer filtering: bilinearly blend four depth-comparison results, not depths. Apply fractional visibility to each light during shading. This reduces jagged edges but does not model an area light's physical penumbra. Keep it independent of material-texture filtering; do not generate depth mipmaps. Clamp directional-map border taps. For emissive lights, remap taps across cube faces with per-face reference depths and average the three comparisons at cube corners.

## Area lights

Use one light list for the configured directional light and emissive triangles from the full, unclipped scene. Emissive triangles remain visible geometry as well as light sources.

Approximate each triangle's illumination with one sample at its world-space centroid, weighted by its area, inverse squared distance, and the receiver and emitter cosines. Emission textures are sampled at the centroid; sidedness follows the material. This is cheap but inaccurate near large triangles and depends on tessellation. Specular highlights also use the centroid, so they do not reproduce the emitter's shape and can be especially inaccurate on smooth surfaces. More samples or area integration can replace it later.

Use rasterized shadow maps at each emitting triangle's centroid: five cube faces aligned to its normal for a one-sided emitter, six for a two-sided emitter. Each sample produces a filtered point-source shadow; this does not integrate visibility over the emitting area or model indirect lighting. Rendering and storage scale with the number of emitting triangles. We keep the per-triangle baseline rather than grouping emitters for now.

Perspective shadow faces store linear light-view depth, normalized by their far distance, with the same one-texel slope bias and configured constant bias. Their range fits the full scene; the near plane is 0.0001 times the far distance, so very nearby occluders can be clipped.

Render direct diffuse and specular lighting, shadows, and surface emission only. Indirect lighting is deferred; its future implementation is undecided. Ignore material occlusion maps, which affect only indirect lighting.

## Transparency

Treat every material as fully opaque. Ignore alpha mode, alpha cutoff, base-color alpha, and texture alpha in camera rendering, shadow maps, and emissive-light collection. Rasterization performs no material texture sampling. Supporting cutouts or blending later requires revisiting visibility and depth ownership.

## Runtime material shading

Use one global material-texture filter: `none` (nearest texel of the nearest mip level), `bilinear` (level zero), or `trilinear` (mipmapped). `none` reads one texel per texture, as an on-sample neural decoder evaluates one texel; it is the initial NTC mode and its baseline, aliasing instead of the noise stochastic filtering would add. Override glTF filter choices but preserve wrap modes. Filter color textures in linear RGB; data maps stay linear. Alpha is ignored. Rasterized passes use analytic perspective-correct UV gradients instead of GPU quad derivatives; non-rasterized point queries use level zero.

Sample all of a material's textures together at one UV before lighting, as an on-sample neural texture decoder produces a whole texture set at once. A material's textures must share one texture-coordinate set; mixed sets are rejected. Fragments carry one UV and gradient pair instead of four, and the normal map is read even when an unusable tangent frame discards it.

Use isotropic box-filtered mipmaps, without anisotropic filtering or normal-variance roughness compensation. Oblique textures can blur. Shadow maps use comparison filtering independently of this setting.

Compute direct diffuse and specular together at runtime, sharing each light's direction and shadow lookup. Use GGX distribution, height-correlated Smith visibility, and Schlick Fresnel, with dielectric F0 = 0.04 and metallic F0 = base color. Multiply direct diffuse reflectance by `1 - F_dielectric(V dot H)`; do not apply the metallic-tinted specular Fresnel to this term. This follows the [glTF metallic-roughness mixing model](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metal-brdf-and-dielectric-brdf), itself an approximation rather than an exactly energy-conserving layered BRDF.

No environment reflections or microfacet multiple-scattering compensation is included. Rough metals can therefore lose energy and metals without directly visible lights can appear dark.

Clamp perceptual roughness to at least 0.045 for specular evaluation, avoiding a singular perfect-mirror lobe and reducing very small highlights. Zero-roughness materials are therefore approximated, not true mirrors.

Preserve the configured directional light's existing brightness convention: intensity 1 gives a facing white Lambertian reference a response of 1 before Fresnel weighting. Convert it to incident irradiance with a factor of pi so diffuse and specular use the same light units.

Prepare shadow maps once per process. Camera movement alone does not require new maps. Disk caching of shadows is deferred.

Keep the framebuffer in linear HDR and apply manual exposure, per-channel Reinhard tone mapping (`x / (1 + x)`), then sRGB encoding only at PNG export. This simple global curve preserves highlight differences but can desaturate bright colors; no automatic exposure or filmic color transform is modeled. Tone mapping can be disabled to recover clamped output.

Use regular-grid supersampling for the CPU reference renderer. A scale of N renders N times the output width and height, with independent coverage, depth, and shading for N² samples per pixel. Resolve color with a box average in linear HDR before tone mapping and sRGB encoding. This reuses the single-sample rasterizer but multiplies camera-framebuffer storage and pixel shading work by N²; it is not an MSAA execution model for GPU traces. Default to 1 (off). Shadow resolution is independent of supersampling.
