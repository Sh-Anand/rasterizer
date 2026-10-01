# Rasterizer notes

Keep these notes updated as we build the rasterizer and revise design decisions.

## Clipping

We currently use explicit triangle clipping instead of a guard band.

- Clip against all six viewing boundaries before perspective division.
- Generate synthetic ("fake") vertices where triangle edges cross a boundary, then triangulate the resulting polygon. These vertices are temporary; the original mesh is unchanged.
- Use cheap accept/reject checks to skip intersection calculations when possible.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.

## Pixel coordinates

After clipping, divide `x`, `y`, and `z` by `w` to obtain normalized device coordinates (NDC), then map to pixel space:

```text
pixel_x = (ndc_x + 1) * width / 2
pixel_y = (1 - ndc_y) * height / 2
depth   = (ndc_z + 1) / 2
```

- Origin is top-left; X increases rightward, Y downward. This Y flip reverses triangle winding relative to NDC.
- Keep subpixel positions as floats, not rounded pixel indices. Image edges are at `0` and `width`/`height`; pixel centers will be at `(x + 0.5, y + 0.5)`.
- Depth is `0` at the near plane and `1` at the far plane, not linear camera-space distance.
- Retain `1/w` for later perspective-correct attribute interpolation. Triangle indices stay unchanged; pixel coverage and depth testing are not implemented yet.
