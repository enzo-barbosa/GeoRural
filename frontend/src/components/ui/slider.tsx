import * as React from 'react'
import { cn } from '@/lib/utils'

interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  valueLabel?: string | number
}

export function Slider({ label, valueLabel, className, ...props }: SliderProps) {
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-sm mb-1">
          <label className="font-medium text-gray-700">{label}</label>
          {valueLabel !== undefined && <span className="text-gray-500">{valueLabel}</span>}
        </div>
      )}
      <input
        type="range"
        className={cn(
          'w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary',
          className
        )}
        {...props}
      />
    </div>
  )
}