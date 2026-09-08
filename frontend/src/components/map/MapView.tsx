import { useEffect } from 'react'
import { MapContainer, TileLayer, LayersControl, useMapEvents } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css'
import L from 'leaflet'
import '@geoman-io/leaflet-geoman-free'

// Fix default marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

// World bounds to prevent infinite zoom out and white space
const WORLD_BOUNDS = [[-90, -180], [90, 180]] as [number, number][]

interface MapViewProps {
  onDraw: (coordinates: number[][][]) => void
  onRemove: () => void
  initialPolygon?: number[][][] | null
  disabled?: boolean
}

function DrawControl({ onDraw, onRemove, initialPolygon, disabled }: MapViewProps) {
  const onPolygonDraw = onDraw
  const onPolygonRemove = onRemove
  const map = useMapEvents({})

  useEffect(() => {
    if (!map) return

    const leafletMap = map as any

    if (initialPolygon && !disabled) {
      const polygon = L.polygon(
        initialPolygon[0].map(([lng, lat]) => [lat, lng]),
        { color: '#2563eb', weight: 2, fillOpacity: 0.1 }
      )
      polygon.addTo(leafletMap)
      leafletMap.pm.addLayer(polygon)
      leafletMap.fitBounds(polygon.getBounds())
    }

    if (disabled) {
      leafletMap.pm.disableDraw('Polygon')
    } else {
      leafletMap.pm.enableDraw('Polygon', {
        cursorMarker: true,
        snappable: false,
      })
    }

    // Handle draw events
    const handleCreate = (e: any) => {
      if (e.layer instanceof L.Polygon) {
        const coordinates = e.layer.getLatLngs().map((ring: any) =>
          ring.map((latlng: any) => [latlng.lng, latlng.lat])
        )
        onDraw(coordinates)
      }
    }

    const handleRemove = () => {
      onRemove()
    }

    leafletMap.on('pm:create', handleCreate)
    leafletMap.on('pm:remove', handleRemove)

    return () => {
      leafletMap.off('pm:create', handleCreate)
      leafletMap.off('pm:remove', handleRemove)
    }
  }, [initialPolygon, disabled, onDraw, onRemove, map])

  return null
}

export function MapView({
  onDraw,
  onRemove,
  initialPolygon,
  disabled = false,
}: MapViewProps) {
  return (
    <div className="w-full h-[500px] rounded-lg border border-gray-200 overflow-hidden" style={{ zIndex: 0 }}>
      <MapContainer
        center={[-14.235, -51.925]}
        zoom={5}
        minZoom={2}
        maxBounds={WORLD_BOUNDS}
        maxBoundsViscosity={1.0}
        worldCopyJump={false}
        style={{ height: '100%', width: '100%' }}
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer name="Satélite" checked>
            <TileLayer
              attribution='&copy; <a href="https://www.esri.com/">Esri</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Ruas">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </LayersControl.BaseLayer>
          <LayersControl.Overlay name="Rótulos (Híbrido)" checked={false}>
            <TileLayer
              attribution='&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://a.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png"
            />
          </LayersControl.Overlay>
        </LayersControl>

        <DrawControl
          onDraw={onDraw}
          onRemove={onRemove}
          initialPolygon={initialPolygon}
          disabled={disabled}
        />
      </MapContainer>
    </div>
  )
}