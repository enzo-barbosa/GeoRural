import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import { Loader2, MapPin, Search } from 'lucide-react'
import { useGeocode } from '@/hooks/useApi'
import { parseCoordinates } from '@/lib/coordinates'
import type { GeocodeResult } from '@/types/api'

export interface LocationSelection {
  lat: number
  lon: number
  bbox?: [number, number, number, number] | null
  displayName?: string
}

interface LocationSearchProps {
  onSelect: (selection: LocationSelection) => void
}

const EMPTY_RESULTS: GeocodeResult[] = []

export function LocationSearch({ onSelect }: LocationSearchProps) {
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(-1)
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const blurTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(query.trim()), 400)
    return () => clearTimeout(timer)
  }, [query])

  useEffect(() => {
    return () => {
      if (blurTimer.current) clearTimeout(blurTimer.current)
    }
  }, [])

  const coordinates = useMemo(() => parseCoordinates(query), [query])
  const { data, isFetching, isError } = useGeocode(coordinates ? '' : debounced)
  const results = data ?? EMPTY_RESULTS

  const choose = (result: GeocodeResult) => {
    onSelect({
      lat: result.lat,
      lon: result.lon,
      bbox: result.bbox ?? null,
      displayName: result.display_name,
    })
    setSelectedName(result.display_name)
    setOpen(false)
    setActive(-1)
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()

    if (coordinates) {
      onSelect({ lat: coordinates.lat, lon: coordinates.lon })
      setSelectedName(
        `Coordenadas: ${coordinates.lat}, ${coordinates.lon}`
      )
      setOpen(false)
      return
    }

    if (results.length > 0) {
      choose(results[active >= 0 ? active : 0])
    }
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (coordinates) return

    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setOpen(true)
      setActive((current) =>
        results.length ? (current + 1) % results.length : -1
      )
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActive((current) =>
        results.length
          ? current <= 0
            ? results.length - 1
            : current - 1
          : -1
      )
    } else if (event.key === 'Escape') {
      setOpen(false)
    }
  }

  const showDropdown = open && !coordinates && debounced.length >= 3

  return (
    <form onSubmit={submit} className="relative mb-3">
      <label htmlFor="location-search" className="sr-only">
        Buscar local
      </label>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
        <input
          id="location-search"
          type="text"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value)
            setSelectedName(null)
            setActive(-1)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => {
            blurTimer.current = setTimeout(() => setOpen(false), 150)
          }}
          onKeyDown={handleKeyDown}
          placeholder="Endereço ou coordenadas (ex.: -22.49, -48.55)"
          className="w-full rounded-md border border-gray-300 py-2 pl-9 pr-20 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          autoComplete="off"
        />
        <button
          type="submit"
          className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:opacity-90"
        >
          Ir
        </button>
      </div>

      {selectedName && (
        <p className="mt-2 flex items-start gap-1 text-xs text-green-700">
          <MapPin className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <span>{selectedName}</span>
        </p>
      )}

      {showDropdown && (
        <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-md border border-gray-200 bg-white shadow-lg">
          {isFetching && (
            <div className="flex items-center gap-2 px-3 py-2 text-sm text-gray-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Buscando...
            </div>
          )}
          {isError && (
            <div className="px-3 py-2 text-sm text-red-600">
              Serviço de endereços indisponível. Use coordenadas (lat, lon).
            </div>
          )}
          {!isFetching && !isError && results.length === 0 && (
            <div className="px-3 py-2 text-sm text-gray-500">
              Nenhum resultado.
            </div>
          )}
          {results.map((result, index) => (
            <button
              key={`${result.lat}-${result.lon}-${index}`}
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(result)}
              className={`block w-full px-3 py-2 text-left text-sm hover:bg-gray-50 ${
                index === active ? 'bg-gray-100' : ''
              }`}
            >
              <span className="block truncate text-gray-800">
                {result.display_name}
              </span>
              <span className="text-xs text-gray-500">
                {result.lat.toFixed(5)}, {result.lon.toFixed(5)}
              </span>
            </button>
          ))}
        </div>
      )}

      <p className="mt-2 text-xs text-gray-500">
        Busque por endereço ou digite{' '}
        <span className="font-medium">latitude, longitude</span>. Depois desenhe
        a área no mapa. Busca por OpenStreetMap/Nominatim.
      </p>
    </form>
  )
}
