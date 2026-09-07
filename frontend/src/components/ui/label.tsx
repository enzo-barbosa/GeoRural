import * as React from 'react'

interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {}

export const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className="block text-sm font-medium text-gray-700 mb-1"
      {...props}
    />
  )
)
Label.displayName = 'Label'