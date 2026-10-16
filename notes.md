# Rasterizer notes

Record design decisions and their tradeoffs, not implementation details or progress.

## GPU backend

Model vertex and fragment shading as distinct programmable stages, with vertex shading intended to run on the GPU. For the initial backend, execute only fragment shading on the GPU; keep vertex processing, rasterization, interpolation, depth/output handling, and the other stages on the CPU. GPU fragment work must consume generated fragments, not scan tile triangle lists or perform software coverage tests. Keep the existing CPU renderer as an independent correctness reference.

Prepare scene geometry independently of the camera. Decode attributes, generate missing flat normals and tangents, and split normal/tangent seams before vertex shading so every backend receives fixed topology and object-space attributes. Vertex shading applies the instance transforms and view projection without changing indices. Keep the existing matrix multiplication and normalization order for the CPU reference.

Select vertex, rasterization, and fragment backends independently. Their initial interfaces exchange host-resident geometry, fragment attributes, and linear HDR colors. Rasterization owns clipping, projection, interpolation, and depth; fragment shading owns material color evaluation. The coordinator allocates the framebuffer and scatters returned colors. Supersampling resolve, exposure, tone mapping, and export remain ordinary framebuffer operations outside the replaceable stages.

Process vertices once for the scene, then resolve visibility across all materials before shading. Retain one final fragment per render-resolution sample, replacing its attributes and material index when a closer fragment wins. This bounds fragment storage by the sample count and eliminates cross-material shading overdraw.

Invoke the fragment stage once for the surviving fragments, skipping empty output. Each fragment carries its material index; shading receives the full material table, with a missing index selecting the default glTF material. The GPU backend uses one dispatch with access to all referenced material and texture data.

Preserve equal-depth winners by processing the default material first, then ascending glTF material index, with stable primitive and triangle order. Strict less-than depth testing keeps the first winner. This traversal order does not create separate shading batches; the CPU reference uses the same order for comparisons.

Each single-sample or SSAA fragment owns one distinct framebuffer sample. The coordinator writes its returned color without further depth arbitration. MSAA would require per-sample ownership masks. This design assumes no shader-side discard or depth replacement.

Shade fragments independently, without quad-based lane exchange. Use analytic perspective-correct UV gradients for texture sampling instead of neighboring fragment values. This avoids helper invocations for UV derivatives, but does not reproduce native quad-packing and helper-lane overhead; shaders needing derivatives of other computed values will need explicit derivative handling.

This staged split is for developing and studying shader workloads, not measuring native graphics-pipeline frame times: CPU work and fragment input/output transfers add overhead that a fully GPU-resident implementation would not have. Treat fragment-to-lane packing as an explicit model, not an exact reproduction of a vendor's hardware scheduler.

Measure stage wall time around each backend call, including transfers and synchronization needed to return host-resident results. Count vertices entering vertex shading and surviving fragments entering fragment shading; SSAA fragments are render-resolution samples. Report scene triangles before clipping, and keep loading, lighting preparation, framebuffer writeback, and output times separate from shader timings.

Sum measured loading, geometry, materials, and shadow preparation as the one-time cost for a static scene. Sum vertex shading, rasterization, fragment shading, render setup/writeback, and postprocessing as the per-render cost when the camera changes. Report PNG encoding and writing separately, outside both totals. These are grouped measurements of the current render, not a steady-state frame-loop benchmark.

## Clipping

Use FP32 for all rendering calculations, including clipping, coverage, interpolation, texture filtering, lighting, and image export.

We chose explicit clipping with synthetic ("fake") intersection vertices instead of a guard band.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.

## Shadows

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

Use one global material-texture filter: `none` (nearest), `bilinear` (level zero), or `trilinear` (mipmapped). Override glTF filter choices but preserve wrap modes. Filter color textures in linear RGB; data maps stay linear. Alpha is ignored. Rasterized passes use analytic perspective-correct UV gradients instead of GPU quad derivatives; non-rasterized point queries use level zero.

Use isotropic box-filtered mipmaps, without anisotropic filtering or normal-variance roughness compensation. Oblique textures can blur. Shadow maps use comparison filtering independently of this setting.

Compute direct diffuse and specular together at runtime, sharing each light's direction and shadow lookup. Use GGX distribution, height-correlated Smith visibility, and Schlick Fresnel, with dielectric F0 = 0.04 and metallic F0 = base color. Multiply direct diffuse reflectance by `1 - F_dielectric(V dot H)`; do not apply the metallic-tinted specular Fresnel to this term. This follows the [glTF metallic-roughness mixing model](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metal-brdf-and-dielectric-brdf), itself an approximation rather than an exactly energy-conserving layered BRDF.

No environment reflections or microfacet multiple-scattering compensation is included. Rough metals can therefore lose energy and metals without directly visible lights can appear dark.

Clamp perceptual roughness to at least 0.045 for specular evaluation, avoiding a singular perfect-mirror lobe and reducing very small highlights. Zero-roughness materials are therefore approximated, not true mirrors.

Preserve the configured directional light's existing brightness convention: intensity 1 gives a facing white Lambertian reference a response of 1 before Fresnel weighting. Convert it to incident irradiance with a factor of pi so diffuse and specular use the same light units.

Prepare shadow maps once per process. Camera movement alone does not require new maps. Disk caching of shadows is deferred.

Keep the framebuffer in linear HDR and apply manual exposure, per-channel Reinhard tone mapping (`x / (1 + x)`), then sRGB encoding only at PNG export. This simple global curve preserves highlight differences but can desaturate bright colors; no automatic exposure or filmic color transform is modeled. Tone mapping can be disabled to recover clamped output.

Use regular-grid supersampling for the CPU reference renderer. A scale of N renders N times the output width and height, with independent coverage, depth, and shading for N² samples per pixel. Resolve color with a box average in linear HDR before tone mapping and sRGB encoding. This reuses the single-sample rasterizer but multiplies camera-framebuffer storage and pixel shading work by N²; it is not an MSAA execution model for GPU traces. Default to 1 (off). Shadow resolution is independent of supersampling.
