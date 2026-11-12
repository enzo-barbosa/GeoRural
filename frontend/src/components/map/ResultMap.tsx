import { ImageOverlay, LayersControl, MapContainer, TileLayer } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import type { LayerInfo } from '@/types/api'

interface ResultMapProps {
  bounds: [[number, number], [number, number]]
  rgbImageUrl?: string | null
  layers: LayerInfo[]
}

export function ResultMap({ bounds, rgbImageUrl, layers }: ResultMapProps) {
  const center: [number, number] = [
    (bounds[0][0] + bounds[1][0]) / 2,
    (bounds[0][1] + bounds[1][1]) / 2,
  ]

  return (
    <div
      className="w-full h-[520px] rounded-lg border border-gray-200 overflow-hidden"
      style={{ zIndex: 0 }}
    >
      <MapContainer
        center={center}
        zoom={15}
        minZoom={2}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />
        {rgbImageUrl && (
          <ImageOverlay url={rgbImageUrl} bounds={bounds} opacity={0.75} />
        )}
        <LayersControl position="topright">
          {layers.map((layer) => (
            <LayersControl.Overlay
              key={layer.name}
              name={layer.label}
              checked={layer.name === 'classification'}
            >
              <ImageOverlay url={layer.image_url} bounds={bounds} opacity={0.75} />
            </LayersControl.Overlay>
          ))}
        </LayersControl>
      </MapContainer>
    </div>
  )
}
