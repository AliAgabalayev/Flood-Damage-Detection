export interface Location {
  id: string;
  name: string;
  center: [number, number];
  bounds: [[number, number], [number, number]];
  scene_date: string;
  model: string;
  flooded_area_km2: number;
  flooded_pct: number;
  mask_url: string | null;
  sar_url: string | null;
  geotiff_url: string | null;
  permanent_water_url: string | null;
}