export function formatDateTime(value: string): string {
  return new Date(value).toLocaleString('pt-BR', {
    dateStyle: 'short',
    timeStyle: 'short',
  })
}

export function formatArea(areaHa: number | null): string {
  if (areaHa == null) return 'Área não calculada'
  return `${areaHa.toLocaleString('pt-BR', {
    maximumFractionDigits: 2,
  })} ha`
}
