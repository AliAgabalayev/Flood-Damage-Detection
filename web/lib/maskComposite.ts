function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// Renders the flood mask with any pixel that's also permanent water made
// transparent, purely for display -- the underlying mask/model data is
// untouched. Keeps "permanent water" and "flood mask" visually mutually
// exclusive even if the two PNGs were produced independently.
export async function suppressPermanentWater(
  floodUrl: string,
  permanentWaterUrl: string,
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

  const floodData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = floodData.data;
  for (let i = 3; i < pixels.length; i += 4) {
    if (pwAlpha[i] > 0) pixels[i] = 0;
  }
  ctx.putImageData(floodData, 0, 0);

  return canvas.toDataURL("image/png");
}
