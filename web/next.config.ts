import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  basePath: process.env.NODE_ENV === "production" ? "/Flood-Damage-Detection" : "",
  allowedDevOrigins: ["*.trycloudflare.com"],
};

export default nextConfig;