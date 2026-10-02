# Rasterizer notes

Record design decisions and their tradeoffs, not implementation details or progress.

## Clipping

We chose explicit clipping with synthetic ("fake") intersection vertices instead of a guard band.

This keeps the implementation straightforward, but can increase vertex and triangle counts. A guard band could later avoid many side-boundary intersections by letting triangles extend beyond the screen and restricting rasterization to visible pixels. It would not eliminate the need to handle triangles crossing the camera plane.
