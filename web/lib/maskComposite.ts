function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// Model detections right at a riverbank often land just outside JRC's exact
// permanent-water footprint (resolution/threshold mismatch, mixed edge
// pixels) -- an exact-pixel suppression leaves a thin flood-colored fringe
// hugging the water, which reads as noise. Dilating the suppression mask by
// a small margin absorbs that fringe. Separable (horizontal pass then
// vertical) so cost is O(w*h*radius), not O(w*h*radius^2).
function dilateAlpha(alpha: Uint8ClampedArray, width: number, height: number, radius: number): Uint8Array {
  const rowMax = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const base = y * width;
    for (let x = 0; x < width; x++) {
      let m = 0;
      const lo = Math.max(0, x - radius);
      const hi = Math.min(width - 1, x + radius);
      for (let xx = lo; xx <= hi; xx++) {
        const a = alpha[(base + xx) * 4 + 3];
        if (a > m) m = a;
      }
      rowMax[base + x] = m;
    }
  }

  const out = new Uint8Array(width * height);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let m = 0;
      const lo = Math.max(0, y - radius);
      const hi = Math.min(height - 1, y + radius);
      for (let yy = lo; yy <= hi; yy++) {
        const v = rowMax[yy * width + x];
        if (v > m) m = v;
      }
      out[y * width + x] = m;
    }
  }
  return out;
}

// Renders the flood mask with any pixel at or near permanent water made
// transparent, purely for display -- the underlying mask/model data is
// untouched. Keeps "permanent water" and "flood mask" visually distinct even
// where the two independently-produced PNGs disagree by a pixel or two at
// the water's edge.
export async function suppressPermanentWater(
  floodUrl: string,
  permanentWaterUrl: string,
  dilationRadius: number = 5,
): Promise<string> {
  const [floodImg, pwImg] = await Promise.all([loadImage(floodUrl), loadImage(permanentWaterUrl)]);

  const pwCanvas = document.createElement("canvas");
  pwCanvas.width = pwImg.width;
  pwCanvas.height = pwImg.height;
  const pwCtx = pwCanvas.getContext("2d");
  if (!pwCtx) return floodUrl;
  pwCtx.drawImage(pwImg, 0, 0);
  const pwAlpha = pwCtx.getImageData(0, 0, pwCanvas.width, pwCanvas.height).data;

  const canvas = document.createElement("canvas");
  canvas.width = floodImg.width;
  canvas.height = floodImg.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return floodUrl;
  ctx.drawImage(floodImg, 0, 0);

  if (canvas.width !== pwCanvas.width || canvas.height !== pwCanvas.height) {
    return floodUrl;
  }

  const suppressMask = dilateAlpha(pwAlpha, canvas.width, canvas.height, dilationRadius);

  const floodData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = floodData.data;
  for (let i = 0, p = 3; p < pixels.length; i++, p += 4) {
    if (suppressMask[i] > 0) pixels[p] = 0;
  }
  ctx.putImageData(floodData, 0, 0);

  return canvas.toDataURL("image/png");
}
