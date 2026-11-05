# Sobel Edge Detection

Beginner | computer-vision | edges | convolution

### The problem, from first principles

An edge is a place where brightness changes quickly, so edge detection is derivative estimation. The **Sobel** operator estimates the horizontal derivative `Gx` and the vertical derivative `Gy` by sliding two `3 x 3` kernels over the image. Each kernel differentiates in one direction and smooths in the other (the weights 1, 2, 1), which makes it less sensitive to noise than a plain difference. The **gradient magnitude** `sqrt(Gx^2 + Gy^2)` is large on edges, and the **direction** `atan2(Gy, Gx)` says which way the brightness changes. This is the hand-designed ancestor of the first convolutional layer of a CNN, which learns kernels like these.

### From theory to code

Implement `sobel_gradients` and `sobel_magnitude`.

### Constraints

- `img` is a 2-D array of shape `(H, W)` with `H, W >= 3`. Use `Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]` and `Ky = Kx.T`.
- `sobel_gradients(img)` returns `(gx, gy)`, each of shape `(H - 2, W - 2)` (valid positions only). Use **cross-correlation** (no kernel flip): `g[i, j] = sum(img[i:i+3, j:j+3] * K)`.
- `sobel_magnitude(img)` returns `sqrt(gx**2 + gy**2)` with the same shape.

### Hints

<details>
<summary>Hint 1</summary>

Sliding the kernel is nine shifted, weighted copies of the image added together, no explicit loops over pixels needed.

</details>

<details>
<summary>Hint 2</summary>

A vertical step edge produces large `gx` and zero `gy`.

</details>
