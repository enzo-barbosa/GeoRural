import { useEffect, useRef } from 'react'
import {
  MapContainer,
  TileLayer,
  LayersControl,
  Marker,
  useMap,
  useMapEvents,
} from 'react-leaflet'
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
  focus?: MapFocus | null
  marker?: [number, number] | null
}

export interface MapFocus {
  lat: number
  lon: number
  zoom?: number
  /** [south, north, west, east], como no Nominatim */
  bbox?: [number, number, number, number] | null
  /** Muda a cada busca para forçar o reposicionamento. */
  id?: number
}

function MapController({
  focus,
  marker,
}: {
  focus?: MapFocus | null
  marker?: [number, number] | null
}) {
  const map = useMap()

  useEffect(() => {
    if (!focus) return
    if (focus.bbox && focus.bbox.length === 4) {
      const [south, north, west, east] = focus.bbox
      map.fitBounds(
        [
          [south, west],
          [north, east],
        ],
        { padding: [24, 24], maxZoom: 16 }
      )
    } else {
      map.flyTo([focus.lat, focus.lon], focus.zoom ?? 14, { duration: 1.2 })
    }
  }, [map, focus])

  if (!marker) return null
  return <Marker position={marker} />
}

function DrawControl({ onDraw, onRemove, initialPolygon, disabled }: MapViewProps) {
  const map = useMapEvents({})

  // Keep the latest callbacks without re-running the effects on every render.
  const onDrawRef = useRef(onDraw)
  const onRemoveRef = useRef(onRemove)
  const polygonRef = useRef<L.Polygon | null>(null)
  const lastInitialRef = useRef<string | null>(null)

  useEffect(() => {
    onDrawRef.current = onDraw
  }, [onDraw])

  useEffect(() => {
    onRemoveRef.current = onRemove
  }, [onRemove])

  // Sync a polygon received from React state (e.g. when restoring the view).
  // The layer created by Geoman on draw is already displayed, so we never
  // recreate it here (that would duplicate the polygon).
  useEffect(() => {
    if (!map) return
    const key = initialPolygon ? JSON.stringify(initialPolygon) : null
    if (key === lastInitialRef.current) return
    lastInitialRef.current = key

    if (!initialPolygon) {
      if (polygonRef.current) {
        polygonRef.current.remove()
        polygonRef.current = null
      }
      return
    }

    if (polygonRef.current) return

    const ring = initialPolygon[0]
    if (!ring || ring.length < 3) return

    const polygon = L.polygon(
      ring.map(([lng, lat]) => [lat, lng]),
      { color: '#2563eb', weight: 2, fillOpacity: 0.1 }
    )
    polygon.addTo(map)
    polygonRef.current = polygon
    map.fitBounds(polygon.getBounds())
  }, [map, initialPolygon])

  // Register Geoman events and toggle draw mode.
  useEffect(() => {
    if (!map) return
    const leafletMap = map as any

    const handleCreate = (e: any) => {
      if (e.layer instanceof L.Polygon) {
        polygonRef.current = e.layer
        const coordinates = e.layer
          .getLatLngs()
          .map((ring: any) => ring.map((latlng: any) => [latlng.lng, latlng.lat]))
        onDrawRef.current(coordinates)
      }
    }

    const handleRemove = () => {
      polygonRef.current = null
      lastInitialRef.current = null
      onRemoveRef.current()
    }

    leafletMap.on('pm:create', handleCreate)
    leafletMap.on('pm:remove', handleRemove)

    if (disabled) {
      leafletMap.pm.disableDraw()
    } else {
      leafletMap.pm.enableDraw('Polygon', {
        cursorMarker: true,
        snappable: false,
      })
    }

    return () => {
      leafletMap.off('pm:create', handleCreate)
      leafletMap.off('pm:remove', handleRemove)
      leafletMap.pm.disableDraw()
    }
  }, [map, disabled])

  return null
}

export function MapView({
  onDraw,
  onRemove,
  initialPolygon,
  disabled = false,
  focus = null,
  marker = null,
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
        <MapController focus={focus} marker={marker} />
      </MapContainer>
    </div>
  )
}
