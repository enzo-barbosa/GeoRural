const COORD_RE =
  /^\s*(-?\d{1,3}(?:\.\d+)?)\s*[\s,;]\s*(-?\d{1,3}(?:\.\d+)?)\s*$/

/**
 * Interpreta "lat, lon" (ou "lon, lat" quando o primeiro valor não é uma
 * latitude válida). Retorna ``null`` quando o texto não é uma coordenada.
 */
export function parseCoordinates(
  input: string
): { lat: number; lon: number } | null {
  const match = COORD_RE.exec(input)
  if (!match) return null

  let lat = Number(match[1])
  let lon = Number(match[2])
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null

  if (Math.abs(lat) > 90 && Math.abs(lon) <= 90) {
    ;[lat, lon] = [lon, lat]
  }
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null

  return { lat, lon }
}
