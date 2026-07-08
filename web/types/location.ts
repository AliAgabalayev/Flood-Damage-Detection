export interface Location {
  id: string;
  name: string;
  center: [number, number];
  bounds: [[number, number], [number, number]];
  scene_date: string;
  model: string;
  flooded_area_km2: number | null;
  flooded_pct: number | null;
  mask_url: string | null;
  sar_url: string | null;
  geotiff_url: string | null;
}