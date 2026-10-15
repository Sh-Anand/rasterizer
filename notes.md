# Rasterizer notes

Record design decisions and their tradeoffs, not implementation details or progress.

## GPU backend

Model vertex and fragment shading as distinct programmable stages, with vertex shading intended to run on the GPU. For the initial backend, execute only fragment shading on the GPU; keep vertex processing, rasterization, interpolation, depth/output handling, and the other stages on the CPU. GPU fragment work must consume generated fragments, not scan tile triangle lists or perform software coverage tests. Keep the existing CPU renderer as an independent correctness reference.

Prepare scene geometry independently of the camera. Decode attributes, generate missing flat normals and tangents, and split normal/tangent seams before vertex shading so every backend receives fixed topology and object-space attributes. Vertex shading applies the instance transforms and view projection without changing indices. Keep the existing matrix multiplication and normalization order for the CPU reference. Splitting tangent seams in object space can change vertex numbering compared with splitting transformed tangents, so invalidate older lightmap layouts that reference those vertices.

Select vertex, rasterization, and fragment backends independently. Their initial interfaces exchange host-resident geometry, fragment attributes, and linear HDR colors. Rasterization owns clipping, projection, interpolation, alpha rejection, and depth; fragment shading owns material color evaluation. The coordinator retains depth across batches and scatters returned colors. Supersampling resolve, exposure, tone mapping, and export remain ordinary framebuffer operations outside the replaceable stages.

Use material-batched forward rendering. For each opaque material batch, the CPU completes vertex processing, rasterization, interpolation, and depth resolution against the accumulated depth buffer before launching GPU fragment shading. Retain the final winning fragment per sample, not every fragment that temporarily passes the depth test: a closer fragment removes that sample from the earlier fragment's ownership.

Process the default material first, followed by ascending glTF material index, preserving primitive and triangle order within a material. Keep strict less-than depth testing: the first sample at an equal depth wins in this material order. Compare the CPU reference using the same order; grouping materials can change equal-depth winners relative to the original scene order.

Surviving fragments have disjoint sample ownership within a launch. Each GPU thread shades one fragment; the CPU copies its returned color to its owned samples without further depth arbitration. With MSAA, ownership is a per-sample mask, and a fragment may retain only part of its original coverage. Drop fragments with no remaining samples. This guarantee assumes no shader-side discard or depth replacement.

Finish a batch's output before processing the next material batch. Later batches may overwrite earlier batches' samples after passing depth testing. Accept this cross-material shading overdraw rather than adding a scene-wide depth prepass or visibility buffer; simplicity takes priority over eliminating wasted shading.

Shade fragments independently, without quad-based lane exchange. Use analytic perspective-correct UV gradients for texture sampling instead of neighboring fragment values. This avoids helper invocations for UV derivatives, but does not reproduce native quad-packing and helper-lane overhead; shaders needing derivatives of other computed values will need explicit derivative handling.

This staged split is for developing and studying shader workloads, not measuring native graphics-pipeline frame times: CPU work and fragment input/output transfers add overhead that a fully GPU-resident implementation would not have. Treat fragment-to-lane packing as an explicit model, not an exact reproduction of a vendor's hardware scheduler.

## Clipping

We chose explicit clipping with synthetic ("fake") intersection vertices instead of a guard band.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.

## Shadows

One orthographic shadow map covers the full scene so off-camera geometry can cast shadows. This keeps fitting simple, but large scenes spread the available resolution thinly. Use a one-texel slope bias to cover the bilinear PCF footprint, plus a configurable constant bias: too little bias causes self-shadowing artifacts; too much detaches shadows from their casters.

Use fixed 2x2 percentage-closer filtering: bilinearly blend four depth-comparison results, not depths. Apply fractional visibility to each light in runtime shading and the indirect bake's direct-light input. This reduces jagged edges but does not model an area light's physical penumbra. Keep it independent of material-texture filtering; do not generate depth mipmaps. Clamp directional-map border taps. For emissive lights, remap taps across cube faces with per-face reference depths and average the three comparisons at cube corners. Filtering changes require rebaking indirect lighting.

## Area lights

Use one light list for the configured directional light and emissive triangles from the full, unclipped scene. Emissive triangles remain visible geometry as well as light sources.

Approximate each triangle's illumination with one sample at its world-space centroid, weighted by its area, inverse squared distance, and the receiver and emitter cosines. Emission textures and alpha masks are also sampled at the centroid; sidedness follows the material. This is cheap but inaccurate near large triangles and depends on tessellation. Specular highlights also use the centroid, so they do not reproduce the emitter's shape and can be especially inaccurate on smooth surfaces. More samples or area integration can replace it later.

Use rasterized shadow maps at each emitting triangle's centroid: five cube faces aligned to its normal for a one-sided emitter, six for a two-sided emitter. Each sample produces a filtered point-source shadow; this does not integrate visibility over the emitting area or model indirect lighting. Rendering and storage scale with the number of emitting triangles. We keep the per-triangle baseline rather than grouping emitters for now.

Perspective shadow faces store linear light-view depth, normalized by their far distance, with the same one-texel slope bias and configured constant bias. Their range fits the full scene; the near plane is 0.0001 times the far distance, so very nearby occluders can be clipped.

Lighting and visibility will remain rasterization-based, including future GI; no ray tracing or shadow rays.

## Baked lighting

Bake exactly one diffuse indirect bounce for static geometry, materials, and lights. The saved lightmap contains no direct lighting. Store linear floating-point indirect lighting for a white receiver; apply the receiver's `base_color * (1 - metallic)` when rendering, alongside runtime direct lighting and emission.

Limit GI to one diffuse bounce to keep CPU baking tractable. Second and later reflections are omitted, so enclosed spaces can be too dark and multi-bounce color bleeding is missing. Increasing bake resolution does not recover these missing light paths.

TODO (deferred): Add configurable indirect-bounce counts, starting with two. Propagate the previous bounce into the next and accumulate only indirect lighting in the saved lightmap. This requires rebaking and increases bake time, but leaves runtime shading unchanged.

Generate a separate lightmap UV atlas with xatlas, giving each placed mesh instance its own space without changing material UVs or source triangle identities. Two-sided surfaces have independent front/back lighting. Chart padding supports bilinear filtering; sub-texel triangles use a centroid sample. The requested resolution is approximate: many small charts can make the atlas substantially larger.

For indirect lighting, rasterize a five-face hemicube at each surface sample and integrate incoming light with cosine/solid-angle weights. Captures read a temporary direct Lambertian lighting atlas, multiplied by the visible surface's `base_color * (1 - metallic)`. Discard the direct atlas after computing the bounce; do not add it to the saved map. This models only diffuse reflection: pure metals contribute no bounce, so specular indirect lighting is deliberately omitted. Exclude emission because emissive triangles already contribute to runtime direct lighting; never feed indirect results back into these captures. Single-sided backfaces block light but do not reflect it.

Keep the initial bake coarse: this CPU method is expensive, and low lightmap/hemicube resolutions can blur shadows, miss small features, and leak light. Cache the result using scene, material, light, and bake settings; camera and image-resolution changes do not require rebaking.

The indirect bake remains Lambertian: both bounce captures and the final receiver omit angle-dependent diffuse Fresnel weighting. Its RGB lightmap does not retain incoming light directions, so removing this approximation would require storing directional lighting and doing more runtime work.

Apply normal maps only to runtime direct diffuse and specular lighting. Keep mesh normals for the indirect bake, including bounce captures; normal-map detail therefore does not affect bounced lighting.

## Transparency

TODO (deferred): Support alpha-blended transparency; `BLEND` materials currently render as opaque. Revisit per-pixel fragment lists for the CPU reference renderer: depth-sort and composite shaded fragments to handle intersecting transparent geometry, at the cost of extra storage and sorting. Transparent shadows and bake visibility also need an explicit policy. No transparency algorithm is selected yet.

## Runtime material shading

Use one global material-texture filter: `none` (nearest), `bilinear` (level zero), or `trilinear` (mipmapped). Override glTF filter choices but preserve wrap modes. Filter color textures in linear RGB; data maps and alpha stay linear. Rasterized passes use analytic perspective-correct UV gradients instead of GPU quad derivatives; non-rasterized point queries use level zero. Changing the filter requires rebaking indirect lighting.

Use isotropic box-filtered mipmaps, without anisotropic filtering, alpha-coverage preservation, or normal-variance roughness compensation. Oblique textures can blur and distant cutouts can thin out. Exclude lightmap-atlas and shadow-depth sampling from this setting; ordinary atlas mipmaps would bleed between charts, and shadows need comparison filtering instead.

Compute direct diffuse and specular together at runtime, sharing each light's direction and shadow lookup. This adds per-pixel diffuse evaluation instead of baking it, but permits camera/light-dependent Fresnel weighting. Use GGX distribution, height-correlated Smith visibility, and Schlick Fresnel, with dielectric F0 = 0.04 and metallic F0 = base color. Multiply direct diffuse reflectance by `1 - F_dielectric(V dot H)`; do not apply the metallic-tinted specular Fresnel to this term. Share the underlying `base_color * (1 - metallic)` reflectance with indirect shading. This follows the [glTF metallic-roughness mixing model](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metal-brdf-and-dielectric-brdf), itself an approximation rather than an exactly energy-conserving layered BRDF.

No specular indirect lighting, environment reflections, or microfacet multiple-scattering compensation is included. Rough metals can therefore lose energy and metals without directly visible lights can appear dark.

Clamp perceptual roughness to at least 0.045 for specular evaluation, avoiding a singular perfect-mirror lobe and reducing very small highlights. Zero-roughness materials are therefore approximated, not true mirrors.

Preserve the configured directional light's existing brightness convention: intensity 1 gives a facing white Lambertian reference a response of 1 before Fresnel weighting. Convert it to incident irradiance with a factor of pi so diffuse and specular use the same light units.

Prepare shadow maps once per process and share them between baking and runtime direct lighting. They remain resident through rendering, increasing memory use during the indirect bake. Cached-lightmap renders still rebuild these maps at startup; camera movement alone does not require new maps or a new indirect bake. Disk caching of shadows is deferred.

Keep the framebuffer in linear HDR and apply manual exposure, per-channel Reinhard tone mapping (`x / (1 + x)`), then sRGB encoding only at PNG export. This simple global curve preserves highlight differences but can desaturate bright colors; no automatic exposure or filmic color transform is modeled. Output settings do not affect the lighting bake. Tone mapping can be disabled to recover clamped output.

Use regular-grid supersampling for the CPU reference renderer. A scale of N renders N times the output width and height, with independent coverage, depth, and shading for N² samples per pixel. Resolve color with a box average in linear HDR before tone mapping and sRGB encoding. This reuses the single-sample rasterizer but multiplies camera-framebuffer storage and pixel shading work by N²; it is not an MSAA execution model for GPU traces. Default to 1 (off). Shadow and bake resolutions stay unchanged, so changing supersampling does not require rebaking.
