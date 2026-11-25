import { LogOut, UserRound } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { useAuth } from '@/contexts/AuthContext'

export function UserMenu() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  if (!user) return null

  const handleLogout = async () => {
    await logout()
    navigate('/')
  }

  return (
    <div className="flex items-center gap-2">
      <span className="hidden sm:flex items-center gap-1.5 text-sm text-gray-600">
        <UserRound className="h-4 w-4" />
        {user.name}
      </span>
      <Button
        variant="ghost"
        size="sm"
        className="gap-1.5"
        onClick={() => {
          void handleLogout()
        }}
      >
        <LogOut className="h-4 w-4" />
        Sair
      </Button>
    </div>
  )
}
